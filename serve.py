"""
Resident TRELLIS.2 process: load the weights once, run many jobs.

Upstream generate.py pays ~100s of weight loading per image. With five tries
per keeper that is most of your wall-clock. This keeps one pipeline in
memory and accepts jobs two ways:

  Watch folder
      python serve.py --watch inbox --out outbox
      Drop a PNG/JPG into inbox/. Output lands in outbox/<name>/.
      Optional sidecar inbox/<name>.json: {"seed": 7, "seeds": [1,2,3],
      "preset": "draft", "pipeline_type": "1024_cascade", "steps": 8,
      "texture_size": 2048, "no_texture": false}

  HTTP
      python serve.py --http 127.0.0.1:8765
      POST /generate  JSON {"image": "/abs/path.png", "seed": 7, ...same keys}
      Multiple seeds in one call: {"image": "...", "seeds": [1, 2, 3]}
      GET /health   -> {"ok": true, "jobs": N}

Both can run together. Jobs are processed one at a time on a single
pipeline; the Metal stack is not safe to share across threads.
"""

from trellis3.runner import bootstrap
bootstrap()  # must precede any torch import

import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image as PILImage

from trellis3 import memory
from trellis3.export import export_mesh, output_stem
from trellis3.presets import apply_preset
from trellis3.runner import Runner, WatchdogError
from trellis3.thermal import wait_until_cool

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_job(spec, image, out_dir):
    """Normalise a job dict into the settings generate() needs."""
    cfg = apply_preset(spec.get("preset"),
                       pipeline_type=spec.get("pipeline_type"),
                       steps=spec.get("steps"),
                       texture_size=spec.get("texture_size"),
                       no_texture=bool(spec.get("no_texture", False)))
    seeds = spec.get("seeds") or [spec.get("seed", 42)]
    seeds = [int(s) for s in seeds]
    stem = os.path.splitext(os.path.basename(image))[0]
    return {
        "image": image,
        "seeds": seeds,
        "out_dir": os.path.join(out_dir, stem),
        "max_num_tokens": int(spec.get("max_tokens", 49152)),
        **cfg,
    }


class JobServer:
    """Owns the Runner and a FIFO queue. `generate_fn` is injectable for tests."""

    def __init__(self, runner_factory, out_dir, thermal_guard=True, log=print):
        self.log = log
        self.out_dir = out_dir
        self.thermal_guard = thermal_guard
        self.queue = queue.Queue()
        self.results = {}
        self.jobs_done = 0
        self._runner_factory = runner_factory
        self._runner = None
        self._lock = threading.Lock()

    @property
    def runner(self):
        with self._lock:
            if self._runner is None:
                self._runner = self._runner_factory()
            return self._runner

    def submit(self, job):
        job_id = f"{int(time.time()*1000)}-{self.jobs_done + self.queue.qsize()}"
        self.results[job_id] = {"status": "queued", "job": job}
        self.queue.put((job_id, job))
        return job_id

    def run_one(self, job_id, job):
        self.results[job_id]["status"] = "running"
        self.log(f"\n=== job {job_id}: {os.path.basename(job['image'])} seeds={job['seeds']} "
                 f"pipeline={job['pipeline_type']} ===")
        if self.thermal_guard:
            wait_until_cool(log=self.log)
        outputs = []
        try:
            img = PILImage.open(job["image"])
            for seed in job["seeds"]:
                t0 = time.time()
                mesh = self.runner.generate(img, seed=seed, pipeline_type=job["pipeline_type"],
                                            steps=job["steps"], max_num_tokens=job["max_num_tokens"])
                stem = output_stem(f"seed{seed}", job["out_dir"])
                paths = export_mesh(mesh, stem, texture_size=job["texture_size"],
                                    no_texture=job["no_texture"], log=self.log)
                paths["seed"] = seed
                paths["seconds"] = round(time.time() - t0, 1)
                outputs.append(paths)
                self.log(f"  seed {seed}: {paths['faces']:,} faces in {paths['seconds']}s")
            self.results[job_id].update(status="done", outputs=outputs)
        except WatchdogError as e:
            self.results[job_id].update(status="failed", error="watchdog", outputs=outputs)
            self.log(str(e))
        except Exception as e:  # keep the server alive on a bad job
            self.results[job_id].update(status="failed", error=f"{type(e).__name__}: {e}",
                                        outputs=outputs)
            self.log(f"  job failed: {type(e).__name__}: {e}")
        finally:
            self.jobs_done += 1
            memory.free_device_cache()
        return self.results[job_id]

    def loop(self, stop_event=None):
        while not (stop_event and stop_event.is_set()):
            try:
                job_id, job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.run_one(job_id, job)


# ---------------------------------------------------------------------------
# Watch folder
# ---------------------------------------------------------------------------

def scan_inbox(inbox, seen):
    """Yield (image_path, sidecar_spec) for new images in inbox."""
    for name in sorted(os.listdir(inbox)):
        path = os.path.join(inbox, name)
        if name.startswith(".") or not name.lower().endswith(IMAGE_EXTS) or path in seen:
            continue
        # skip files still being written
        if time.time() - os.path.getmtime(path) < 1.0:
            continue
        spec = {}
        sidecar = os.path.splitext(path)[0] + ".json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar) as f:
                    spec = json.load(f)
            except (OSError, ValueError) as e:
                print(f"  bad sidecar {sidecar}: {e}; using defaults")
        seen.add(path)
        yield path, spec


def watch_loop(server, inbox, out_dir, stop_event, poll=2.0):
    seen = set()
    os.makedirs(inbox, exist_ok=True)
    while not stop_event.is_set():
        for path, spec in scan_inbox(inbox, seen):
            job = parse_job(spec, path, out_dir)
            jid = server.submit(job)
            print(f"  queued {os.path.basename(path)} as {jid}")
        stop_event.wait(poll)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(server, out_dir):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True, "jobs": server.jobs_done,
                                        "queued": server.queue.qsize()})
            if self.path.startswith("/job/"):
                jid = self.path[len("/job/"):]
                res = server.results.get(jid)
                return self._json(200 if res else 404, res or {"error": "unknown job"})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                spec = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            image = spec.get("image")
            if not image or not os.path.exists(image):
                return self._json(400, {"error": "image path missing or not found"})
            try:
                job = parse_job(spec, image, out_dir)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            jid = server.submit(job)
            self._json(202, {"job_id": jid, "status": "queued"})

        def log_message(self, fmt, *args):  # quiet
            pass

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Resident TRELLIS.2 job server")
    p.add_argument("--watch", help="Inbox folder to poll for images")
    p.add_argument("--http", help="host:port for the JSON API, e.g. 127.0.0.1:8765")
    p.add_argument("--out", default="outbox", help="Output folder (default: outbox)")
    p.add_argument("--pipeline-types", default="512",
                   help="Comma-separated types to preload (others load lazily on first use)")
    p.add_argument("--resident", action="store_true", help="Keep models on the GPU (low_vram=False)")
    p.add_argument("--no-thermal-guard", action="store_true")
    args = p.parse_args(argv)
    if not args.watch and not args.http:
        p.error("give --watch and/or --http")

    types = [t.strip() for t in args.pipeline_types.split(",") if t.strip()]
    server = JobServer(lambda: Runner(types, resident=args.resident), args.out,
                       thermal_guard=not args.no_thermal_guard)
    server.runner  # load now, not on first job

    stop = threading.Event()
    threads = [threading.Thread(target=server.loop, args=(stop,), daemon=True)]
    if args.watch:
        threads.append(threading.Thread(target=watch_loop, args=(server, args.watch, args.out, stop),
                                        daemon=True))
    httpd = None
    if args.http:
        host, port = args.http.rsplit(":", 1)
        httpd = ThreadingHTTPServer((host, int(port)), make_handler(server, args.out))
        threads.append(threading.Thread(target=httpd.serve_forever, daemon=True))
    for t in threads:
        t.start()
    print(f"\nReady. watch={args.watch or '-'} http={args.http or '-'} out={args.out}. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        if httpd:
            httpd.shutdown()
        print("stopped")


if __name__ == "__main__":
    sys.exit(main())
