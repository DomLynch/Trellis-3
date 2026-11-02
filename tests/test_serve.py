import json
import os
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch
from PIL import Image

import serve
from trellis3.runner import WatchdogError


class _FakeMesh:
    def __init__(self):
        self.vertices = torch.zeros((3, 3))
        self.faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.attrs = None


class _FakeRunner:
    def __init__(self, fail_seed=None):
        self.calls = []
        self.fail_seed = fail_seed

    def generate(self, image, seed, pipeline_type, steps, max_num_tokens):
        self.calls.append((seed, pipeline_type, steps, max_num_tokens))
        if seed == self.fail_seed:
            raise WatchdogError("boom")
        return _FakeMesh()


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "cat.png"
    Image.new("RGB", (8, 8), "red").save(p)
    return str(p)


@pytest.fixture
def fake_export(monkeypatch):
    written = []

    def _export(mesh, stem, texture_size=1024, no_texture=False, log=print, **kw):
        written.append((stem, texture_size, no_texture))
        return {"glb": stem + ".glb", "obj": stem + ".obj", "faces": 1, "vertices": 3, "bake": "none"}

    monkeypatch.setattr(serve, "export_mesh", _export)
    return written


def test_parse_job_defaults(png, tmp_path):
    job = serve.parse_job({}, png, str(tmp_path / "out"))
    assert job["seeds"] == [42] and job["pipeline_type"] == "512"
    assert job["out_dir"].endswith("out/cat")


def test_parse_job_seeds_and_preset(png, tmp_path):
    job = serve.parse_job({"seeds": [1, "2"], "preset": "draft", "max_tokens": 1000}, png, "o")
    assert job["seeds"] == [1, 2] and job["steps"] == 8 and job["no_texture"] is True
    assert job["max_num_tokens"] == 1000


def test_run_one_seed_sweep(png, tmp_path, fake_export):
    runner = _FakeRunner()
    srv = serve.JobServer(lambda: runner, str(tmp_path / "out"), thermal_guard=False,
                          log=lambda *_: None)
    job = serve.parse_job({"seeds": [1, 2, 3], "pipeline_type": "1024_cascade", "steps": 6}, png,
                          str(tmp_path / "out"))
    jid = srv.submit(job)
    res = srv.run_one(jid, job)
    assert res["status"] == "done" and len(res["outputs"]) == 3
    assert [c[0] for c in runner.calls] == [1, 2, 3]
    from trellis3.presets import default_max_tokens
    assert runner.calls[0][1:] == ("1024_cascade", 6, default_max_tokens())
    assert all(s.startswith(str(tmp_path / "out" / "cat")) for s, _, _ in fake_export)
    assert srv.jobs_done == 1


def test_run_one_survives_failure(png, tmp_path, fake_export):
    runner = _FakeRunner(fail_seed=2)
    srv = serve.JobServer(lambda: runner, str(tmp_path), thermal_guard=False, log=lambda *_: None)
    job = serve.parse_job({"seeds": [1, 2, 3]}, png, str(tmp_path))
    res = srv.run_one(srv.submit(job), job)
    assert res["status"] == "failed" and res["error"] == "watchdog"
    assert len(res["outputs"]) == 1  # seed 1 completed before the failure
    # server still works afterwards
    job2 = serve.parse_job({}, png, str(tmp_path))
    assert srv.run_one(srv.submit(job2), job2)["status"] == "done"


def test_runner_is_created_lazily_once(tmp_path):
    made = []
    srv = serve.JobServer(lambda: made.append(1) or _FakeRunner(), str(tmp_path), log=lambda *_: None)
    assert made == []
    srv.runner; srv.runner
    assert made == [1]


def test_scan_inbox_with_sidecar(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    Image.new("RGB", (4, 4)).save(inbox / "a.png")
    (inbox / "a.json").write_text('{"seed": 9}')
    (inbox / "notes.txt").write_text("x")
    (inbox / ".hidden.png").write_text("x")
    old = time.time() - 5
    for f in inbox.iterdir():
        os.utime(f, (old, old))
    seen = set()
    found = list(serve.scan_inbox(str(inbox), seen))
    assert len(found) == 1
    assert found[0][0].endswith("a.png") and found[0][1] == {"seed": 9}
    assert list(serve.scan_inbox(str(inbox), seen)) == []  # not re-queued


def test_scan_inbox_skips_fresh_files(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "new.png")
    assert list(serve.scan_inbox(str(tmp_path), set())) == []


def test_http_roundtrip(png, tmp_path, fake_export):
    runner = _FakeRunner()
    srv = serve.JobServer(lambda: runner, str(tmp_path / "out"), thermal_guard=False,
                          log=lambda *_: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(srv, str(tmp_path / "out")))
    port = httpd.server_address[1]
    stop = threading.Event()
    threads = [threading.Thread(target=httpd.serve_forever, daemon=True),
               threading.Thread(target=srv.loop, args=(stop,), daemon=True)]
    for t in threads:
        t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/health") as r:
            assert json.load(r)["ok"] is True

        req = urllib.request.Request(base + "/generate", method="POST",
                                     data=json.dumps({"image": png, "seeds": [4, 5]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 202
            jid = json.load(r)["job_id"]

        for _ in range(100):
            with urllib.request.urlopen(f"{base}/job/{jid}") as r:
                res = json.load(r)
            if res["status"] == "done":
                break
            time.sleep(0.05)
        assert res["status"] == "done" and [o["seed"] for o in res["outputs"]] == [4, 5]

        bad = urllib.request.Request(base + "/generate", method="POST",
                                     data=b'{"image": "/nope.png"}',
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        stop.set()
        httpd.shutdown()


def test_parse_job_bake_options(png):
    job = serve.parse_job({"metallic_factor": -1, "gamma": 2.2, "fix_winding": False,
                           "uv_cone_deg": 120}, png, "o")
    assert job["bake"] == {"metallic_factor": None, "gamma": 2.2, "fix_winding": False,
                           "uv_cone_deg": 120, "uv_smooth": None, "normal_map": False,
                           "normal_flip_v": False, "snap_hires": True, "bake_faces": 200_000,
                           "bake_sampling": "trilinear"}
    assert serve.parse_job({}, png, "o")["bake"]["metallic_factor"] == 0.0


def test_wait_for_memory(monkeypatch):
    monkeypatch.setattr(serve.time, "sleep", lambda s: None)
    seq = iter([2.0, 3.0, 9.0])
    ok, free = serve.wait_for_memory(8.0, max_wait=60, log=lambda *_: None, probe=lambda: next(seq))
    assert ok and free == 9.0
    assert serve.wait_for_memory(0, 60) == (True, None)
    assert serve.wait_for_memory(8.0, 60, probe=lambda: None) == (True, None)  # unknown never blocks
    t = iter([0, 0, 100, 200, 300])
    monkeypatch.setattr(serve.time, "time", lambda: next(t))
    ok, free = serve.wait_for_memory(8.0, max_wait=50, log=lambda *_: None, probe=lambda: 1.0)
    assert not ok and free == 1.0


def test_run_one_skips_job_on_low_memory(png, tmp_path, fake_export, monkeypatch):
    monkeypatch.setattr(serve, "wait_for_memory", lambda *a, **k: (False, 1.5))
    srv = serve.JobServer(lambda: _FakeRunner(), str(tmp_path), thermal_guard=False,
                          min_free_gb=8, log=lambda *_: None)
    job = serve.parse_job({}, png, str(tmp_path))
    res = srv.run_one(srv.submit(job), job)
    assert res["status"] == "failed" and "low memory" in res["error"]
    assert srv.jobs_done == 1


def test_run_one_writes_receipts(png, tmp_path, fake_export):
    srv = serve.JobServer(lambda: _FakeRunner(), str(tmp_path / "out"), thermal_guard=False,
                          log=lambda *_: None)
    job = serve.parse_job({"seeds": [5], "gamma": 2.2}, png, str(tmp_path / "out"))
    res = srv.run_one(srv.submit(job), job)
    r = json.load(open(res["outputs"][0]["receipt"]))
    assert r["settings"]["seed"] == 5 and r["settings"]["bake"]["gamma"] == 2.2
    assert r["job_id"].endswith("-1") and r["input"]["size"] == [8, 8]
