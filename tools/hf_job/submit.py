"""Package this repo + inputs into the private dataset repo and print the `hf jobs run` command (or run it).

    python tools/hf_job/submit.py wheels --go                 # one-off CPU job that builds the CUDA wheels
    python tools/hf_job/submit.py hero --run head1 --inputs view_0.png view_1.png ... [--flavor a100-large] [--gen-args "..."] --go
Without --go nothing billed starts. The job credential is forwarded by name as a job secret and never printed.
The code bundle is `git archive <ref>`: pass the multi-image branch with this branch's tools/hf_job overlaid (see docs/hf-gpu-jobs.md).
"""
import argparse
import os
import subprocess
import sys
import tempfile

IO_REPO = os.environ.get("HF_IO_REPO", "Domlynch/trellis-hero-io")
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HERO = "--preset hero --single-process --max-tokens 98304 --fusion multidiffusion --seed 42"
SCRIPTS = ("build_wheels.sh", "run_hero.sh")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["wheels", "hero"])
    p.add_argument("--ref", default="HEAD", help="git ref of the code to bundle (needs generate.py --images)")
    p.add_argument("--run", default="head1")
    p.add_argument("--inputs", nargs="*", default=[])
    p.add_argument("--flavor", default=None)
    p.add_argument("--gen-args", default=HERO)
    p.add_argument("--timeout", default=None)
    p.add_argument("--go", action="store_true")
    a = p.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(IO_REPO, repo_type="dataset", private=True, exist_ok=True)
    for n in SCRIPTS:
        api.upload_file(path_or_fileobj=os.path.join(HERE, n), path_in_repo=f"bundle/{n}", repo_id=IO_REPO, repo_type="dataset")
    if a.what == "hero":
        with tempfile.TemporaryDirectory() as d:
            tar = os.path.join(d, "trellis3.tar")
            subprocess.check_call(["git", "-C", REPO, "archive", "-o", tar, a.ref])
            subprocess.check_call(["gzip", "-f", tar])
            api.upload_file(path_or_fileobj=tar + ".gz", path_in_repo="bundle/trellis3.tar.gz", repo_id=IO_REPO, repo_type="dataset")
        for f in a.inputs:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"runs/{a.run}/in/{os.path.basename(f)}", repo_id=IO_REPO, repo_type="dataset")
    script = "build_wheels.sh" if a.what == "wheels" else "run_hero.sh"
    boot = (f"pip install -q huggingface_hub && python -c \"from huggingface_hub import hf_hub_download as d;"
            f"d('{IO_REPO}','bundle/{script}',repo_type='dataset',local_dir='/boot')\" && bash /boot/bundle/{script}")
    cmd = ["hf", "jobs", "run", "--flavor", a.flavor or ("cpu-xl" if a.what == "wheels" else "a100-large"),
           "--timeout", a.timeout or ("90m" if a.what == "wheels" else "3h"), "-s", "HF_TOKEN",
           "-e", f"IO_REPO={IO_REPO}", "-e", f"RUN_NAME={a.run}", "-e", f"GEN_ARGS={a.gen_args}",
           "--name", f"trellis-{a.what}-{a.run}", "-d", IMAGE, "bash", "-lc", boot]
    print(" ".join(repr(c) if " " in c else c for c in cmd))
    if a.go:
        sys.exit(subprocess.call(cmd))


main()
