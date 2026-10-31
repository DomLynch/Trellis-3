from trellis3 import thermal

PMSET_HOT = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
CPU_Scheduler_Limit 	= 100
CPU_Available_CPUs 	= 10
CPU_Speed_Limit 	= 54
"""


def test_parse_speed_limit():
    assert thermal.parse_speed_limit(PMSET_HOT) == 54
    assert thermal.parse_speed_limit("CPU_Speed_Limit = 100") == 100
    assert thermal.parse_speed_limit("") is None
    assert thermal.parse_speed_limit(None) is None


def test_cpu_speed_limit_without_pmset(monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("pmset")

    monkeypatch.setattr(subprocess, "run", boom)
    assert thermal.cpu_speed_limit() is None


def test_wait_until_cool_returns_when_recovered(monkeypatch):
    seq = iter([40, 60, 90])
    monkeypatch.setattr(thermal, "cpu_speed_limit", lambda: next(seq))
    monkeypatch.setattr(thermal.time, "sleep", lambda s: None)
    logs = []
    assert thermal.wait_until_cool(min_speed=85, max_wait=60, poll=1, log=logs.append) == 90
    assert any("waiting" in l for l in logs)


def test_wait_until_cool_never_blocks_without_data(monkeypatch):
    monkeypatch.setattr(thermal, "cpu_speed_limit", lambda: None)
    assert thermal.wait_until_cool(log=lambda *_: None) is None


def test_wait_until_cool_gives_up(monkeypatch):
    monkeypatch.setattr(thermal, "cpu_speed_limit", lambda: 30)
    monkeypatch.setattr(thermal.time, "sleep", lambda s: None)
    t = iter([0, 0, 100, 200])
    monkeypatch.setattr(thermal.time, "time", lambda: next(t))
    assert thermal.wait_until_cool(max_wait=50, poll=1, log=lambda *_: None) == 30
