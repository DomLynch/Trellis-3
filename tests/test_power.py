from trellis3 import power


def test_refuses_on_battery_only():
    assert power.refuse_on_battery(probe=lambda: True)
    assert power.refuse_on_battery(probe=lambda: False) is None
    assert power.refuse_on_battery(probe=lambda: None) is None


def test_override_flag_and_env(monkeypatch):
    assert power.refuse_on_battery(allow=True, probe=lambda: True) is None
    monkeypatch.setenv("TRELLIS3_ALLOW_BATTERY", "1")
    assert power.refuse_on_battery(probe=lambda: True) is None
