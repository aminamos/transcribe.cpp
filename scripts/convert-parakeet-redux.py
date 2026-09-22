#!/usr/bin/env python3
"""convert-parakeet-redux.py - convert moondream/parakeet-redux (thrush-ternary
packed safetensors, HF module names) into a GGUF that transcribe.cpp's
parakeet loader ingests end-to-end.

The redux checkpoint is a ternary-quantized parakeet-tdt-0.6b-v3: same
architecture, tokenizer, and output conventions; every encoder linear
weight is packed 5 base-3 digits per byte (format "thrush-ternary-v2",
see ternary.json in the source repo). This script dequantizes those
modules exactly - w = scales[row, col // 128] * (code - 1) - into fp32
and reuses convert-parakeet.py's emission pipeline (name tables, KV
metadata, tensor layout) by presenting the weights as a NeMo-shaped
state_dict. No NeMo install is required.

    uv run --no-project --with numpy --with gguf --with omegaconf \
      python scripts/convert-parakeet-redux.py \
        /path/to/parakeet-redux-dir \
        models/parakeet-redux/parakeet-redux-F32.gguf

The tokenizer payload (tokens/scores/types) is copied verbatim from the
already-shipped handy-computer parakeet-tdt-0.6b-v3 GGUF header -
identical tokenizer per the redux model card - via --ref-gguf (the
header is fetched over HTTP range when given a URL).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import struct
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

REF_GGUF_URL = (
    "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf"
    "/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf"
)

# ---------------------------------------------------------------------------
# Minimal safetensors reader (numpy-only, mmap-backed)
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Parse a .safetensors file into a name -> numpy array dict.

    Only the dtypes the redux checkpoint actually carries are mapped;
    BF16 is deliberately absent (not present in the file, and numpy has
    no native bfloat16).
    """
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    (hdr_len,) = struct.unpack("<Q", bytes(raw[:8]))
    hdr = json.loads(bytes(raw[8 : 8 + hdr_len]))
    base = 8 + hdr_len
    out: dict[str, np.ndarray] = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        dt = _ST_DTYPES.get(info["dtype"])
        if dt is None:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        b0, b1 = info["data_offsets"]
        arr = np.frombuffer(
            bytes(raw[base + b0 : base + b1]), dtype=dt
        ).reshape(info["shape"])
        out[name] = arr
    return out


# ---------------------------------------------------------------------------
# thrush-ternary-v2 dequantization
# ---------------------------------------------------------------------------


def dequant_ternary(
    qweight: np.ndarray,
    scales: np.ndarray,
    out_features: int,
    in_features: int,
    group_size: int = 128,
) -> np.ndarray:
    """Unpack a packed ternary weight to fp32.

    Packing (ternary.json): row-major, 5 base-3 digits per byte, element i
    of a row is digit (i % 5) of byte (i // 5), least-significant digit
    first, row padded to ceil(in/5) bytes. Reconstruction:
    w[row, col] = scales[row, col // group_size] * (code - 1), code in
    {0, 1, 2}.
    """
    row_bytes = (in_features + 4) // 5
    if qweight.shape != (out_features, row_bytes):
        raise ValueError(
            f"qweight shape {qweight.shape} != ({out_features}, {row_bytes})"
        )
    if scales.shape != (out_features, in_features // group_size):
        raise ValueError(
            f"scales shape {scales.shape} != "
            f"({out_features}, {in_features // group_size})"
        )

    b = qweight.astype(np.uint16)
    codes = np.empty((out_features, row_bytes * 5), dtype=np.float32)
    for j in range(5):
        codes[:, j::5] = ((b // (3**j)) % 3).astype(np.float32)
    codes -= 1.0  # {0,1,2} -> {-1,0,+1}; padding digits land past in_features
    codes = codes[:, :in_features]

    s = scales.astype(np.float32)
    return codes * np.repeat(s, group_size, axis=1)


# ---------------------------------------------------------------------------
# TQ1_0 emission (ggml block_tq1_0, 256 elems/block, 1.6875 bpw)
# ---------------------------------------------------------------------------
#
# Redux stores one fp16 scale per 128-element group; block_tq1_0 carries one
# fp16 scale per 256-element block, so each block merges a pair of redux
# groups. For each block we pick the merged scale d by trying two candidates
# and keeping whichever decodes with lower squared error:
#
#   keep-all:  d = min(d*, 2*s_min) — every trit survives (a trit dies only
#              when |w|/d < 0.5; larger quotients saturate to +-1 via clip)
#   kill-weak: d = s_max            — strong group exact, weak group -> 0
#
# where d* is the nonzero-count-weighted mean of the two source scales
# (least-squares optimal merged scale under a preserved trit pattern). On
# this checkpoint the merge lands at ~0.10-0.23 relative L2 error per
# tensor, worst on the attention k/v/o projections.
#
# Packing matches ggml's quantize_row_tq1_0_ref / dequantize_row_tq1_0:
#   qs[48]: 5 trits/byte MSB-first, byte = ceil(q * 256/243)
#           bytes  0..31 hold elems m + n*32  (m<32, n<5)   -> elems   0..159
#           bytes 32..47 hold elems 160+m+n*16 (m<16, n<5)  -> elems 160..239
#   qh[4]:  byte j holds elems 240+j+4m (m<4), group shifted up one trit (*3)
#   d:     trailing fp16

def pack_tq1_0(w: np.ndarray, scales: np.ndarray, group_size: int = 128) -> np.ndarray:
    """[out, in] fp32 weights + [out, in/group_size] fp32/fp16 scales ->
    packed block_tq1_0 bytes as a [out, (in/256)*54] uint8 array."""
    out_f, in_f = w.shape
    if in_f % 256 != 0:
        raise ValueError(f"in={in_f} not divisible by TQ1_0 block 256")
    nb = in_f // 256
    wb = w.reshape(out_f, nb, 2, 128).astype(np.float64)
    s = scales.astype(np.float64).reshape(out_f, nb, 2)

    s0, s1 = s[..., 0], s[..., 1]
    nz0 = (wb[:, :, 0, :] != 0).sum(-1)
    nz1 = (wb[:, :, 1, :] != 0).sum(-1)
    nz = nz0 + nz1

    smax = np.maximum(s0, s1)
    smin = np.minimum(s0, s1)
    # A trit survives iff d16 <= 2*|w| (round(|w|/d) >= 1; clip saturates
    # oversized quotients to +-1 rather than killing them). Per block the
    # choice is between:
    #   keep-all:  d = min(d*, 2*smin)  preserves every trit
    #   kill-weak: d = smax             strong group exact, weak trits -> 0
    # where d* is the nz-weighted mean (least-squares optimal merged scale).
    # Score both candidates by actually decoding against the fp16-rounded
    # scale and summing squared error over the block.
    d_star = np.where(nz > 0, (nz0 * s0 + nz1 * s1) / np.maximum(nz, 1), smax)
    cand = np.stack([np.minimum(d_star, 2.0 * smin), smax], -1)
    cand16 = cand.astype(np.float16).astype(np.float64)
    # fp16 rounding may push the keep candidate above 2*smin, which would
    # silently kill the weak group — back off to the largest fp16 below.
    below = np.nextafter((2.0 * smin).astype(np.float16),
                         np.float16(0)).astype(np.float64)
    cand16[..., 0] = np.where(cand16[..., 0] > 2.0 * smin,
                              below, cand16[..., 0])
    w5 = wb.reshape(out_f, nb, 256)
    scaled = w5[..., None] / np.maximum(cand16[..., None, :], 1e-30)
    qq = np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + 0.5), -1, 1)
    err = ((qq * cand16[:, :, None, :] - w5[..., None]) ** 2).sum(2)
    pick = np.argmin(err, axis=-1)                       # [out, nb]
    d16 = np.take_along_axis(cand16, pick[..., None], -1)[..., 0]
    q = np.take_along_axis(qq, pick[..., None, None], -1)[..., 0]
    t = (q + 1).astype(np.uint16)                          # {-1,0,1} -> {0,1,2}

    def pack5(v):  # MSB-first base-3 group -> byte
        return ((v * 256 + 242) // 243).astype(np.uint8)

    b1 = (t[:, :, 0:32] * 81 + t[:, :, 32:64] * 27 + t[:, :, 64:96] * 9
          + t[:, :, 96:128] * 3 + t[:, :, 128:160])        # [out, nb, 32]
    b2 = (t[:, :, 160:176] * 81 + t[:, :, 176:192] * 27 + t[:, :, 192:208] * 9
          + t[:, :, 208:224] * 3 + t[:, :, 224:240])       # [out, nb, 16]
    bh = ((t[:, :, 240:244] * 27 + t[:, :, 244:248] * 9
           + t[:, :, 248:252] * 3 + t[:, :, 252:256]) * 3)  # [out, nb, 4]

    block = np.concatenate(
        [pack5(b1), pack5(b2), pack5(bh),
         d16.astype(np.float16).view(np.uint8).reshape(out_f, nb, 2)],
        axis=-1)
    return np.ascontiguousarray(block.reshape(out_f, nb * 54))


# ---------------------------------------------------------------------------
# TQ1_128 emission (ggml block_tq1_128, 128 elems/block, 1.75 bpw)
# ---------------------------------------------------------------------------
#
# block_tq1_128 is the native form of thrush-ternary-v2: one fp16 scale per
# 128-element group and the same 5-trits-per-byte, LSB-first packing. Each
# 128-group gets its own 26 bytes (the source packs a whole row contiguously,
# so group boundaries straddle bytes), and within a block the trits are
# plane-interleaved — byte j digit l holds element 25*l + j (j < 25), byte 25
# holds elements 125..127 — so a SIMD decoder sees contiguous activation
# runs per digit plane. Repacking is exact: codes and scales transfer
# verbatim, no scale merging.

# ggml type id for block_tq1_128 (GGML_TYPE_TQ1_128 in ggml.h); gguf-py
# doesn't know it, so the byte-shape -> logical-shape table gets an entry.
GGML_TYPE_TQ1_128 = 43
TQ1_128_QUANT_SIZES = (128, 28)  # (block elems, block bytes) = 26 qs + 2 d

# Downstream general.file_type tag (llama file types end at ~42; no upstream
# value exists for this type). Informational only — nothing validates it.
FILE_TYPE_TQ1_128 = 43


def pack_tq1_128(qweight: np.ndarray, scales: np.ndarray,
                 out_features: int, in_features: int,
                 group_size: int = 128) -> np.ndarray:
    """[out, ceil(in/5)] source qweight + [out, in/gs] scales ->
    packed block_tq1_128 bytes as a [out, (in/128)*28] uint8 array."""
    if in_features % group_size != 0:
        raise ValueError(f"in={in_features} not divisible by {group_size}")
    nb = in_features // group_size
    row_bytes = (in_features + 4) // 5
    if qweight.shape != (out_features, row_bytes):
        raise ValueError(f"qweight shape {qweight.shape}")

    b = qweight.astype(np.uint16)
    codes = np.empty((out_features, row_bytes * 5), dtype=np.uint8)
    for j in range(5):
        codes[:, j::5] = (b // (3 ** j)) % 3
    codes = codes[:, :in_features].reshape(out_features, nb, group_size)

    pow3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
    qs = np.empty((out_features, nb, 26), dtype=np.uint8)
    # bytes 0..24: digit l holds element 25*l + j (plane-interleaved)
    planes = codes[:, :, :125].reshape(out_features, nb, 5, 25)
    qs[:, :, :25] = (planes * pow3[None, None, :, None]).sum(axis=2)
    # byte 25: digits 0..2 hold elements 125..127
    qs[:, :, 25] = (codes[:, :, 125:128] * pow3[:3]).sum(-1)

    d = scales.astype(np.float16).view(np.uint8).reshape(out_features, nb, 2)
    return np.ascontiguousarray(
        np.concatenate([qs, d], axis=-1).reshape(out_features, nb * 28))


# ---------------------------------------------------------------------------
# redux (HF-style names) -> NeMo state_dict names
# ---------------------------------------------------------------------------
#
# The redux module tree mirrors HF's parakeet_tdt layout; NeMo names the
# same tensors differently. convert-parakeet.py's emission tables are
# written against NeMo names, so we translate once up front and feed the
# pipeline a state_dict it already understands.

_REDUX_ATTN = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_out",
    "relative_k_proj": "linear_pos",
}


def build_nemo_state_dict(
    sd: dict[str, np.ndarray], ternary_meta: dict
) -> dict[str, np.ndarray]:
    """Return NeMo-named fp32 tensors from a redux safetensors dict."""
    out: dict[str, np.ndarray] = {}

    quantized = {m["name"]: m for m in ternary_meta["quantized_modules"]}
    group_size = int(ternary_meta["quant"]["group_size"])

    def dequant(name: str) -> np.ndarray:
        m = quantized[name]
        w = dequant_ternary(
            sd[f"{name}.qweight"],
            sd[f"{name}.scales"],
            m["out_features"],
            m["in_features"],
            group_size,
        )
        if m.get("as_conv1d"):
            # quantized as a 1x1 Conv1d: [out, in] -> [out, in, 1]
            w = w.reshape(m["out_features"], m["in_features"], 1)
        return w

    # --- encoder pre_encode (subsampling conv stack) ---
    # redux: encoder.subsampling.layers.{0,2,3,5,6} (convs) + .linear
    for n in ("0", "2", "3", "5", "6"):
        for kind in ("weight", "bias"):
            out[f"encoder.pre_encode.conv.{n}.{kind}"] = np.ascontiguousarray(
                sd[f"encoder.subsampling.layers.{n}.{kind}"].astype(np.float32)
            )
    out["encoder.pre_encode.out.weight"] = np.ascontiguousarray(
        sd["encoder.subsampling.linear.weight"].astype(np.float32)
    )
    out["encoder.pre_encode.out.bias"] = np.ascontiguousarray(
        sd["encoder.subsampling.linear.bias"].astype(np.float32)
    )

    # --- encoder layers ---
    n_layers = 0
    while f"encoder.layers.{n_layers}.norm_out.weight" in sd:
        n_layers += 1

    for i in range(n_layers):
        p = f"encoder.layers.{i}"

        # layernorms (unquantized fp32)
        for norm, short in (
            ("norm_feed_forward1", "norm_feed_forward1"),
            ("norm_self_att", "norm_self_att"),
            ("norm_conv", "norm_conv"),
            ("norm_feed_forward2", "norm_feed_forward2"),
            ("norm_out", "norm_out"),
        ):
            for kind in ("weight", "bias"):
                out[f"{p}.{norm}.{kind}"] = np.ascontiguousarray(
                    sd[f"{p}.{short}.{kind}"].astype(np.float32)
                )

        # macaron feed-forwards (ternary)
        for ff in ("feed_forward1", "feed_forward2"):
            for lin in ("linear1", "linear2"):
                out[f"{p}.{ff}.{lin}.weight"] = dequant(f"{p}.{ff}.{lin}")

        # self-attention (ternary projections + fp16 pos biases)
        for redux_name, nemo_name in _REDUX_ATTN.items():
            out[f"{p}.self_attn.{nemo_name}.weight"] = dequant(
                f"{p}.self_attn.{redux_name}"
            )
        out[f"{p}.self_attn.pos_bias_u"] = np.ascontiguousarray(
            sd[f"{p}.self_attn.bias_u"].astype(np.float32)
        )
        out[f"{p}.self_attn.pos_bias_v"] = np.ascontiguousarray(
            sd[f"{p}.self_attn.bias_v"].astype(np.float32)
        )

        # conv module: pointwise convs are ternary; depthwise is fp16
        out[f"{p}.conv.pointwise_conv1.weight"] = dequant(
            f"{p}.conv.pointwise_conv1"
        )
        out[f"{p}.conv.pointwise_conv2.weight"] = dequant(
            f"{p}.conv.pointwise_conv2"
        )
        out[f"{p}.conv.depthwise_conv.weight"] = np.ascontiguousarray(
            sd[f"{p}.conv.depthwise_conv.weight"].astype(np.float32)
        )
        # redux conv.norm == NeMo conv.batch_norm
        for kind in ("weight", "bias", "running_mean", "running_var"):
            out[f"{p}.conv.batch_norm.{kind}"] = np.ascontiguousarray(
                sd[f"{p}.conv.norm.{kind}"].astype(np.float32)
            )

    # --- predictor (decoder) ---
    out["decoder.prediction.embed.weight"] = np.ascontiguousarray(
        sd["decoder.embedding.weight"].astype(np.float32)
    )
    i = 0
    while f"decoder.lstm.weight_ih_l{i}" in sd:
        out[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}"] = (
            np.ascontiguousarray(
                sd[f"decoder.lstm.weight_ih_l{i}"].astype(np.float32)
            )
        )
        out[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}"] = (
            np.ascontiguousarray(
                sd[f"decoder.lstm.weight_hh_l{i}"].astype(np.float32)
            )
        )
        out[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}"] = (
            np.ascontiguousarray(
                sd[f"decoder.lstm.bias_ih_l{i}"].astype(np.float32)
            )
        )
        out[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}"] = (
            np.ascontiguousarray(
                sd[f"decoder.lstm.bias_hh_l{i}"].astype(np.float32)
            )
        )
        i += 1

    # --- joint ---
    out["joint.enc.weight"] = np.ascontiguousarray(
        sd["encoder_projector.weight"].astype(np.float32)
    )
    out["joint.enc.bias"] = np.ascontiguousarray(
        sd["encoder_projector.bias"].astype(np.float32)
    )
    out["joint.pred.weight"] = np.ascontiguousarray(
        sd["decoder.decoder_projector.weight"].astype(np.float32)
    )
    out["joint.pred.bias"] = np.ascontiguousarray(
        sd["decoder.decoder_projector.bias"].astype(np.float32)
    )
    out["joint.joint_net.2.weight"] = np.ascontiguousarray(
        sd["joint.head.weight"].astype(np.float32)
    )
    out["joint.joint_net.2.bias"] = np.ascontiguousarray(
        sd["joint.head.bias"].astype(np.float32)
    )

    return out


# ---------------------------------------------------------------------------
# Reference-GGUF tokenizer extraction
# ---------------------------------------------------------------------------


def fetch_range(url: str, nbytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def read_gguf_kvs(buf: bytes) -> dict:
    """Parse just the KV section of a GGUF header."""
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    _ver, _n_tensors, n_kv = struct.unpack("<IQQ", buf[4:24])
    off = 24

    def rd_str(o: int):
        (n,) = struct.unpack("<Q", buf[o : o + 8])
        return buf[o + 8 : o + 8 + n].decode("utf-8"), o + 8 + n

    kvs: dict = {}
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
            10: "Q", 11: "q", 12: "d"}
    for _ in range(n_kv):
        k, off = rd_str(off)
        (t,) = struct.unpack("<I", buf[off : off + 4])
        off += 4
        if t == 8:
            v, off = rd_str(off)
            kvs[k] = v
        elif t == 9:
            et, n = struct.unpack("<IQ", buf[off : off + 12])
            off += 12
            if et == 8:
                arr = []
                for _ in range(n):
                    s, off = rd_str(off)
                    arr.append(s)
                kvs[k] = arr
            else:
                kvs[k] = list(
                    struct.unpack("<" + fmts[et] * n, buf[off : off + sizes[et] * n])
                )
                off += sizes[et] * n
        else:
            kvs[k] = struct.unpack("<" + fmts[t], buf[off : off + sizes[t]])[0]
            off += sizes[t]
    return kvs


def load_ref_tokenizer(ref_gguf: str) -> dict:
    """Tokenizer payload from the shipped parakeet-tdt-0.6b-v3 GGUF.

    Same tokenizer per the redux model card; copying the shipped arrays
    keeps the emitted tokenizer.ggml.* bytes identical to what the
    runtime already ingests.
    """
    p = Path(ref_gguf)
    if p.exists():
        head = p.read_bytes()[: 8 * 1024 * 1024]
    else:
        head = fetch_range(ref_gguf, 8 * 1024 * 1024)
    kvs = read_gguf_kvs(head)
    tokens = kvs["tokenizer.ggml.tokens"]
    scores = kvs["tokenizer.ggml.scores"]
    types_ = kvs["tokenizer.ggml.token_type"]
    return {
        "tokens": tokens,
        "scores": [float(s) for s in scores],
        "types": [int(t) for t in types_],
        "unk_id": int(kvs.get("tokenizer.ggml.unknown_token_id", 0)),
        "bos_id": None,  # v3 ships neither bos nor eos ids
        "eos_id": None,
        "blank_id": int(kvs["tokenizer.ggml.blank_token_id"]),
    }


# ---------------------------------------------------------------------------
# Fake NeMo-model facade for convert-parakeet.py
# ---------------------------------------------------------------------------


class _ReduxModel:
    """The surface convert-parakeet.py's convert() consumes:

    .cfg (omegaconf node), .state_dict(), .tokenizer.tokenizer
    .serialized_model_proto(), .eval(), .encoder/.joint = None (falls
    back to cfg + state_dict shapes).
    """

    def __init__(self, cfg_dict: dict, sd: dict[str, np.ndarray]):
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.create(cfg_dict)
        self._sd = sd
        self.encoder = None
        self.joint = None

        class _Tok:
            class tokenizer:  # noqa: D106 - attribute shim
                @staticmethod
                def serialized_model_proto() -> bytes:
                    return b""

        self.tokenizer = _Tok()

    def state_dict(self):
        return self._sd

    def eval(self):
        return self


# NeMo-shaped model_config equivalent for parakeet-tdt-0.6b-v3, verified
# against the shipped handy-computer GGUF header.
_REDUX_CFG = {
    "encoder": {
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 8,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "pos_emb_max_len": 5000,
        "xscaling": False,
        "att_context_size": [-1, -1],
        "conv_norm_type": "batch_norm",
        "conv_context_size": None,
    },
    "decoder": {"vocab_size": 8192, "prednet": {}},
    "joint": {
        "jointnet": {
            "joint_hidden": 640,
            "activation": "relu",
            "num_extra_outputs": 5,
        }
    },
    "decoding": {
        "durations": [0, 1, 2, 3, 4],
        "greedy": {"max_symbols": 10},
    },
    "preprocessor": {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "features": 128,
        "sample_rate": 16000,
        "n_fft": 512,
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "normalize": "per_feature",
        "dither": 1e-5,
    },
}


# ---------------------------------------------------------------------------
# TQ1_0 pre-pack wiring
# ---------------------------------------------------------------------------
#
# Redux module tail -> gguf encoder-block suffix. The pointwise convs are
# packed too — they are k=1 (i.e. plain matmuls) and are emitted flat as
# [in, out] TQ1_0 rather than the [1, in, out] conv-kernel layout; the
# parakeet loader accepts both forms for conv.pointwise{1,2}.weight.

_REDUX_TO_GGUF_SUFFIX = {
    "feed_forward1.linear1": "ff1.linear1.weight",
    "feed_forward1.linear2": "ff1.linear2.weight",
    "feed_forward2.linear1": "ff2.linear1.weight",
    "feed_forward2.linear2": "ff2.linear2.weight",
    "self_attn.q_proj": "attn.linear_q.weight",
    "self_attn.k_proj": "attn.linear_k.weight",
    "self_attn.v_proj": "attn.linear_v.weight",
    "self_attn.o_proj": "attn.linear_out.weight",
    "self_attn.relative_k_proj": "attn.linear_pos.weight",
    "conv.pointwise_conv1": "conv.pointwise1.weight",
    "conv.pointwise_conv2": "conv.pointwise2.weight",
}


def build_tq1_blobs(sd: dict[str, np.ndarray], ternary_meta: dict) -> dict[str, np.ndarray]:
    """{gguf_name: packed block_tq1_0 uint8 rows} for every 2-D ternary module."""
    import re

    blobs: dict[str, np.ndarray] = {}
    group_size = int(ternary_meta["quant"]["group_size"])
    for m in ternary_meta["quantized_modules"]:
        mm = re.fullmatch(r"encoder\.layers\.(\d+)\.(.*)", m["name"])
        if mm is None or mm.group(2) not in _REDUX_TO_GGUF_SUFFIX:
            continue
        i, tail = int(mm.group(1)), mm.group(2)
        w = dequant_ternary(
            sd[f"{m['name']}.qweight"], sd[f"{m['name']}.scales"],
            m["out_features"], m["in_features"], group_size)
        blobs[f"enc.blocks.{i}.{_REDUX_TO_GGUF_SUFFIX[tail]}"] = pack_tq1_0(
            w, sd[f"{m['name']}.scales"], group_size)
    return blobs


def build_tq1_128_blobs(sd: dict[str, np.ndarray], ternary_meta: dict) -> dict[str, np.ndarray]:
    """{gguf_name: packed block_tq1_128 uint8 rows} — exact repack of the
    source ternary codes/scales, no dequantize->requantize round trip."""
    import re

    blobs: dict[str, np.ndarray] = {}
    group_size = int(ternary_meta["quant"]["group_size"])
    for m in ternary_meta["quantized_modules"]:
        mm = re.fullmatch(r"encoder\.layers\.(\d+)\.(.*)", m["name"])
        if mm is None or mm.group(2) not in _REDUX_TO_GGUF_SUFFIX:
            continue
        i, tail = int(mm.group(1)), mm.group(2)
        blobs[f"enc.blocks.{i}.{_REDUX_TO_GGUF_SUFFIX[tail]}"] = pack_tq1_128(
            sd[f"{m['name']}.qweight"], sd[f"{m['name']}.scales"],
            m["out_features"], m["in_features"], group_size)
    return blobs


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("redux_dir", type=Path,
                   help="directory holding model.safetensors + ternary.json")
    p.add_argument("out_path", type=Path,
                   help="output GGUF path; parent dir name must be the "
                        "variant slug (e.g. models/parakeet-redux/)")
    p.add_argument("--ref-gguf", default=REF_GGUF_URL,
                   help="path or URL of the shipped parakeet-tdt-0.6b-v3 "
                        "GGUF used for the tokenizer payload")
    p.add_argument("--tq1", action="store_true",
                   help="emit the 2-D ternary encoder linears as pre-packed "
                        "TQ1_0 blocks (optimal merged per-256 scale) instead "
                        "of fp32; everything else stays fp32")
    p.add_argument("--tq1-128", action="store_true",
                   help="emit the ternary encoder tensors as pre-packed "
                        "TQ1_128 blocks — the native ggml form of "
                        "thrush-ternary-v2 (per-128 scales, no merge); "
                        "everything else stays fp32")
    args = p.parse_args(argv[1:])

    st_path = args.redux_dir / "model.safetensors"
    tj_path = args.redux_dir / "ternary.json"
    for f in (st_path, tj_path):
        if not f.exists():
            print(f"error: missing {f}", file=sys.stderr)
            return 2

    ternary_meta = json.loads(tj_path.read_text())
    if ternary_meta.get("format") != "thrush-ternary-v2":
        print(f"error: unsupported ternary format "
              f"{ternary_meta.get('format')!r}", file=sys.stderr)
        return 2

    print(f"Loading {st_path} ...")
    sd = load_safetensors(st_path)
    print(f"  {len(sd)} tensors; dequantizing ternary modules ...")
    nemo_sd = build_nemo_state_dict(sd, ternary_meta)
    print(f"  {len(nemo_sd)} NeMo-named tensors reconstructed")

    tok = load_ref_tokenizer(args.ref_gguf)
    if len(tok["tokens"]) != 8193:
        print(f"error: ref tokenizer has {len(tok['tokens'])} tokens, "
              f"expected 8193", file=sys.stderr)
        return 2

    # --- drive convert-parakeet.py ---
    spec = importlib.util.spec_from_file_location(
        "convert_parakeet", SCRIPT_DIR / "convert-parakeet.py"
    )
    cpm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpm)

    model = _ReduxModel(_REDUX_CFG, nemo_sd)
    cpm.load_nemo_model = lambda spec_, prefer_direct=False: model
    cpm.tensor_to_fp32_numpy = lambda t: np.ascontiguousarray(
        np.asarray(t, dtype=np.float32)
    )
    cpm.extract_tokenizer = lambda sp, blank_piece="<blank>": tok

    if args.tq1:
        import gguf as _gguf

        blobs = build_tq1_blobs(sd, ternary_meta)
        print(f"  {len(blobs)} tensors pre-packed as TQ1_0 "
              f"({sum(b.nbytes for b in blobs.values()) / 1e6:.1f} MB)")

        orig_gw = cpm.gguf_writer

        def gw(path, arch, **kw):
            w = orig_gw(path, arch, **kw)
            orig_at = w.add_tensor

            def at(name, arr, **kw2):
                blob = blobs.get(name)
                if blob is not None:
                    return orig_at(
                        name, blob,
                        raw_dtype=_gguf.GGMLQuantizationType.TQ1_0)
                return orig_at(name, arr, **kw2)

            w.add_tensor = at
            return w

        cpm.gguf_writer = gw

        # file_type metadata: 36 = MOSTLY_TQ1_0 in the llama.cpp ftype
        # enum mirrored by tools/transcribe-quantize.
        orig_agi = cpm.add_general_identity

        def agi(w, **kw):
            kw["file_type"] = 36
            return orig_agi(w, **kw)

        cpm.add_general_identity = agi

    if args.tq1_128:
        import gguf as _gguf
        from gguf.quants import quantize as _gguf_quantize

        _gguf.GGML_QUANT_SIZES[GGML_TYPE_TQ1_128] = TQ1_128_QUANT_SIZES

        blobs = build_tq1_128_blobs(sd, ternary_meta)
        print(f"  {len(blobs)} tensors pre-packed as TQ1_128 "
              f"({sum(b.nbytes for b in blobs.values()) / 1e6:.1f} MB)")

        # Emit the final layout directly — no transcribe-quantize pass.
        # The TQ1_128 policy preset would ternarize the *dense* predictor/
        # joint weights too (their ne0=640 divides the 128-block), which
        # destroys them. Instead: redux tensors -> packed TQ1_128, dense
        # 2-D weights -> Q8_0 (same treatment as the TQ1_0 build's
        # fallback), conv kernels -> F16, norms/biases stay F32.
        orig_gw = cpm.gguf_writer

        def gw(path, arch, **kw):
            w = orig_gw(path, arch, **kw)
            orig_at = w.add_tensor

            def at(name, arr, **kw2):
                blob = blobs.get(name)
                if blob is not None:
                    return orig_at(name, blob, raw_dtype=GGML_TYPE_TQ1_128)
                if (arr.ndim == 2 and name.endswith(".weight")
                        and arr.shape[-1] % 32 == 0):
                    q8 = _gguf_quantize(
                        np.ascontiguousarray(arr, dtype=np.float32),
                        _gguf.GGMLQuantizationType.Q8_0)
                    return orig_at(
                        name, q8,
                        raw_dtype=_gguf.GGMLQuantizationType.Q8_0)
                if arr.ndim >= 3:
                    return orig_at(
                        name, np.ascontiguousarray(arr, dtype=np.float16),
                        **kw2)
                return orig_at(name, arr, **kw2)

            w.add_tensor = at
            return w

        cpm.gguf_writer = gw

        orig_agi = cpm.add_general_identity

        def agi(w, **kw):
            kw["file_type"] = FILE_TYPE_TQ1_128
            return orig_agi(w, **kw)

        cpm.add_general_identity = agi

    # convert() constructs an spm.SentencePieceProcessor purely to feed
    # extract_tokenizer; both are stubbed above, so a no-op shim replaces
    # the module unconditionally — a real sentencepiece install would
    # reject the empty serialized proto our model shim returns.
    fake_spm = types.ModuleType("sentencepiece")

    class _SP:
        def LoadFromSerializedProto(self, _b):
            pass

    fake_spm.SentencePieceProcessor = _SP
    sys.modules["sentencepiece"] = fake_spm

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    cpm.convert(str(st_path), args.out_path, repo_id="moondream/parakeet-redux")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
