import torch
import torch.nn.functional as F

from backends import attention_mps as am


def _ref_pad(q, k, v, q_seqlen, kv_seqlen):
    """Upstream's loop padding."""
    B, H = len(q_seqlen), q.shape[-2]
    mq, mkv = max(q_seqlen), max(kv_seqlen)
    qp = torch.zeros(B, mq, H, q.shape[-1]); kp = torch.zeros(B, mkv, H, k.shape[-1])
    vp = torch.zeros(B, mkv, H, v.shape[-1])
    qo = kvo = 0
    for b in range(B):
        ql, kvl = q_seqlen[b], kv_seqlen[b]
        qp[b, :ql] = q[qo:qo + ql]; kp[b, :kvl] = k[kvo:kvo + kvl]; vp[b, :kvl] = v[kvo:kvo + kvl]
        qo += ql; kvo += kvl
    return qp.permute(0, 2, 1, 3), kp.permute(0, 2, 1, 3), vp.permute(0, 2, 1, 3)


def test_pad_uniform_is_a_view_and_equals_reference():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2 * 7, 3, 8, generator=g); k = torch.randn(2 * 7, 3, 8, generator=g)
    v = torch.randn(2 * 7, 3, 5, generator=g)
    qp, kp, vp = am.pad_uniform(q, k, v, [7, 7], [7, 7])
    rq, rk, rv = _ref_pad(q, k, v, [7, 7], [7, 7])
    assert torch.equal(qp, rq) and torch.equal(kp, rk) and torch.equal(vp, rv)
    assert qp._base is q or qp.data_ptr() == q.data_ptr()     # no copy


def test_pad_ragged_matches_reference():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(5 + 3, 2, 4, generator=g); k = torch.randn(6 + 2, 2, 4, generator=g)
    v = torch.randn(6 + 2, 2, 4, generator=g)
    got = am.pad_uniform(q, k, v, [5, 3], [6, 2])
    ref = _ref_pad(q, k, v, [5, 3], [6, 2])
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    out = torch.randn(2, 2, 5, 4, generator=g)
    un = am.unpad(out, [5, 3])
    assert un.shape == (8, 2, 4)
    assert torch.equal(un[:5], out[0].permute(1, 0, 2)) and torch.equal(un[5:], out[1, :, :3].permute(1, 0, 2))


def test_sdpa_chunked_is_exact(monkeypatch):
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 4, 50, 16, generator=g); k = torch.randn(1, 4, 50, 16, generator=g)
    v = torch.randn(1, 4, 50, 16, generator=g)
    full = F.scaled_dot_product_attention(q, k, v)
    for chunk in (1, 7, 49, 50, 500):
        out = am.sdpa_chunked(F.scaled_dot_product_attention, q, k, v, chunk=chunk)
        assert torch.allclose(out, full, atol=1e-6), chunk
    monkeypatch.setenv("ATTN_CHUNK", "8")
    assert am.chunk_size() == 8
    assert torch.allclose(am.sdpa_chunked(F.scaled_dot_product_attention, q, k, v), full, atol=1e-6)
    monkeypatch.setenv("ATTN_CHUNK", "junk")
    assert am.chunk_size() == 0


def test_end_to_end_equals_unpadded_attention():
    """pad -> chunked sdpa -> unpad on a uniform batch equals plain attention."""
    g = torch.Generator().manual_seed(3)
    T, H, C = 30, 2, 8
    q = torch.randn(T, H, C, generator=g); k = torch.randn(T, H, C, generator=g); v = torch.randn(T, H, C, generator=g)
    qp, kp, vp = am.pad_uniform(q, k, v, [T], [T])
    out = am.unpad(am.sdpa_chunked(F.scaled_dot_product_attention, qp, kp, vp, chunk=4), [T])
    ref = F.scaled_dot_product_attention(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None],
                                         v.permute(1, 0, 2)[None])[0].permute(1, 0, 2)
    assert torch.allclose(out, ref, atol=1e-6)


def test_sdpa_tiled_is_exact_for_awkward_sizes():
    g = torch.Generator().manual_seed(4)
    for (L, S, qt, kt) in ((50, 50, 7, 11), (33, 70, 33, 70), (1, 9, 4, 4), (64, 17, 16, 5), (100, 100, 1000, 1000)):
        q = torch.randn(2, 3, L, 16, generator=g); k = torch.randn(2, 3, S, 16, generator=g)
        v = torch.randn(2, 3, S, 8, generator=g)
        ref = F.scaled_dot_product_attention(q, k, v)
        out = am.sdpa_tiled(q, k, v, qt, kt)
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.allclose(out, ref, atol=1e-5), (L, S, qt, kt)


def test_sdpa_tiled_bf16_inputs_and_large_logits():
    g = torch.Generator().manual_seed(5)
    q = (torch.randn(1, 2, 40, 32, generator=g) * 8).to(torch.bfloat16)   # big logits: tests the running max
    k = (torch.randn(1, 2, 40, 32, generator=g) * 8).to(torch.bfloat16)
    v = torch.randn(1, 2, 40, 32, generator=g).to(torch.bfloat16)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = am.sdpa_tiled(q, k, v, 16, 16)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref, atol=3e-2)


def test_dispatcher_prefers_tiles_over_chunk(monkeypatch):
    g = torch.Generator().manual_seed(6)
    q = torch.randn(1, 2, 30, 8, generator=g); k = torch.randn(1, 2, 30, 8, generator=g); v = torch.randn(1, 2, 30, 8, generator=g)
    ref = F.scaled_dot_product_attention(q, k, v)
    calls = []
    def spy(a, b, c):
        calls.append(a.shape[2]); return F.scaled_dot_product_attention(a, b, c)
    monkeypatch.setenv("ATTN_TILE", "8,8"); monkeypatch.setenv("ATTN_CHUNK", "5")
    assert am.tile_sizes() == (8, 8)
    out = am.sdpa_chunked(spy, q, k, v)
    assert torch.allclose(out, ref, atol=1e-5) and calls == []          # tiled path, sdpa_fn unused
    monkeypatch.setenv("ATTN_TILE", "")
    out = am.sdpa_chunked(spy, q, k, v)
    assert torch.allclose(out, ref, atol=1e-6) and calls == [5] * 6       # chunked path
    monkeypatch.setenv("ATTN_TILE", "junk")
    assert am.tile_sizes() is None
    monkeypatch.setenv("ATTN_TILE", "2048x4096")
    assert am.tile_sizes() == (2048, 4096)
