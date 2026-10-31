"""
Helpers for the SDPA attention path on MPS (installed into
TRELLIS.2/trellis2/modules/sparse/attention/t3_sdpa.py by the patch script).

Two things the upstream trellis-mac SDPA branch does that cost memory on the
49K-token 1024 cascade:

1. It always pads variable-length sequences into a fresh [B, max_len, H, C]
   buffer with a Python loop, even when there is one sequence (B=1, the
   normal case). `pad_uniform` returns a zero-copy view when every sequence
   has the same length, and only falls back to the padded copy otherwise.

2. It runs one SDPA call over the full query length. Whether that
   materialises the [H, L, S] score matrix depends on which kernel PyTorch's
   MPS backend picks for the dtype/head-dim; when it does, 12 heads x 49K x
   49K is tens of GB. `sdpa_chunked` splits the query dimension into chunks
   of ATTN_CHUNK rows (env var, default 0 = off), bounding that term to
   H x chunk x S. Exact: softmax over keys is per query row, so chunking
   queries changes nothing numerically.
"""

import os

import torch


def pad_uniform(q, k, v, q_seqlen, kv_seqlen, device=None):
    """Return (q, k, v) as [B, H, L, C] tensors.

    q: [T_Q, H, C], k/v: [T_KV, H, C]. Zero-copy when all sequences in a
    batch have the same length; otherwise pads like upstream.
    """
    B = len(q_seqlen)
    if kv_seqlen is None:
        kv_seqlen = q_seqlen
    H = q.shape[-2]
    if len(set(q_seqlen)) == 1 and len(set(kv_seqlen)) == 1:
        Lq, Lkv = q_seqlen[0], kv_seqlen[0]
        qp = q.reshape(B, Lq, H, q.shape[-1])
        kp = k.reshape(B, Lkv, H, k.shape[-1])
        vp = v.reshape(B, Lkv, H, v.shape[-1])
    else:
        max_q, max_kv = max(q_seqlen), max(kv_seqlen)
        qp = torch.zeros(B, max_q, H, q.shape[-1], device=q.device, dtype=q.dtype)
        kp = torch.zeros(B, max_kv, H, k.shape[-1], device=k.device, dtype=k.dtype)
        vp = torch.zeros(B, max_kv, H, v.shape[-1], device=v.device, dtype=v.dtype)
        qo = kvo = 0
        for b in range(B):
            ql, kvl = q_seqlen[b], kv_seqlen[b]
            qp[b, :ql] = q[qo:qo + ql]
            kp[b, :kvl] = k[kvo:kvo + kvl]
            vp[b, :kvl] = v[kvo:kvo + kvl]
            qo += ql
            kvo += kvl
    return qp.permute(0, 2, 1, 3), kp.permute(0, 2, 1, 3), vp.permute(0, 2, 1, 3)


def unpad(out, q_seqlen):
    """[B, H, L, C] -> [T_Q, H, C] dropping padding rows."""
    out = out.permute(0, 2, 1, 3)
    if len(set(q_seqlen)) == 1 and out.shape[1] == q_seqlen[0]:
        return out.reshape(-1, out.shape[2], out.shape[3])
    return torch.cat([out[b, :q_seqlen[b]] for b in range(len(q_seqlen))], dim=0)


def chunk_size():
    try:
        return int(os.environ.get("ATTN_CHUNK", "0"))
    except ValueError:
        return 0


def tile_sizes():
    """ATTN_TILE='Lq,Sk' -> (Lq, Sk); unset/invalid -> None."""
    raw = os.environ.get("ATTN_TILE", "")
    if not raw:
        return None
    try:
        a, b = (int(x) for x in raw.replace("x", ",").split(","))
        return (a, b) if a > 0 and b > 0 else None
    except ValueError:
        return None


def sdpa_tiled(q, k, v, q_tile=2048, k_tile=4096, scale=None):
    """Exact attention with memory bounded to [B, H, q_tile, k_tile].

    Flash-attention arithmetic in plain PyTorch: for each query tile, walk
    the keys in tiles keeping a running max and running sum (online softmax)
    in fp32, so the full [L, S] score matrix never exists. On MPS, where
    scaled_dot_product_attention materialises that matrix for these shapes
    (12 heads x 17.7K^2 x 4 B = 15 GB in the profiled 1024 run), this is the
    difference between fitting in 24 GB and not. Slower than a fused kernel:
    a Python loop of (L/q_tile) x (S/k_tile) small matmuls.
    q/k/v: [B, H, L|S, C]. Returns [B, H, L, Cv] in q's dtype.
    """
    B, H, L, C = q.shape
    S = k.shape[2]
    Cv = v.shape[3]
    scale = (C ** -0.5) if scale is None else scale
    out = torch.empty(B, H, L, Cv, dtype=q.dtype, device=q.device)
    for qs in range(0, L, q_tile):
        qb = q[:, :, qs:qs + q_tile].float() * scale                     # [B,H,lq,C]
        m = torch.full((B, H, qb.shape[2], 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qb.shape[2], 1), device=q.device)
        acc = torch.zeros((B, H, qb.shape[2], Cv), device=q.device)
        for ks in range(0, S, k_tile):
            kb = k[:, :, ks:ks + k_tile].float()
            vb = v[:, :, ks:ks + k_tile].float()
            s = qb @ kb.transpose(-1, -2)                                # [B,H,lq,sk]
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            p = torch.exp(s - m_new)
            alpha = torch.exp(m - m_new)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + p @ vb
            m = m_new
        out[:, :, qs:qs + q_tile] = (acc / l).to(q.dtype)
    return out


def sdpa_chunked(sdpa_fn, q, k, v, chunk=None, tiles=None):
    """Attention dispatcher. q/k/v: [B, H, L|S, C].

    ATTN_TILE=Lq,Sk  -> sdpa_tiled (bounded memory, exact, slower)
    ATTN_CHUNK=N     -> sdpa_fn over query chunks (bounds one axis only)
    otherwise        -> sdpa_fn as is
    """
    tiles = tile_sizes() if tiles is None else tiles
    if tiles:
        return sdpa_tiled(q, k, v, *tiles)
    chunk = chunk_size() if chunk is None else chunk
    L = q.shape[2]
    if chunk <= 0 or chunk >= L:
        return sdpa_fn(q, k, v)
    outs = []
    for s in range(0, L, chunk):
        outs.append(sdpa_fn(q[:, :, s:s + chunk], k, v))
    return torch.cat(outs, dim=2)
