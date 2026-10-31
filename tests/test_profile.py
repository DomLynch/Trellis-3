import time

from trellis3 import memory, profile


def test_profiler_attributes_peaks_to_stages():
    seq = {"v": 1.0}
    prof = profile.Profiler(interval=0.005, readers={"fake_gb": lambda: seq["v"]})
    prof.start("load")
    time.sleep(0.03)
    seq["v"] = 5.0
    prof.mark("sample")
    time.sleep(0.03)
    seq["v"] = 2.0
    prof.mark("decode")
    time.sleep(0.03)
    rep = prof.stop()
    names = [s["stage"] for s in rep["stages"]]
    assert names == ["load", "sample", "decode"]
    peaks = {s["stage"]: s["peak"]["fake_gb"] for s in rep["stages"]}
    assert peaks["sample"] == 5.0 and peaks["decode"] == 2.0
    assert peaks["load"] <= 5.0
    assert rep["overall_peak"]["fake_gb"] == 5.0
    assert "sample" in prof.table()


def test_stage_hooks_mark_profiler():
    marks = []

    class P:
        def sample_tex_slat(self, x):
            return x

    class Fake:
        def mark(self, n):
            marks.append(n)

    profile.activate(Fake())
    try:
        p = P()
        memory.install_stage_hooks(p, after=lambda: None)
        p.sample_tex_slat(1)
    finally:
        profile.activate(None)
    assert marks == ["sample_tex_slat", "after:sample_tex_slat"]


def test_rss_reader_works_or_none():
    v = profile._rss_now_gb()
    assert v is None or v > 0
