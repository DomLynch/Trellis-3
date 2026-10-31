"""
One JSON receipt per run: settings, input, outputs with hashes, timings,
peak memory, environment. Written next to the GLB as <output>.receipt.json.
Downstream tooling reads it instead of parsing logs.
"""

import hashlib
import json
import os
import platform
import sys
import time


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        pass
    try:
        from trellis3 import __version__
        out["trellis3"] = __version__
    except Exception:
        pass
    return out


def write_receipt(output, settings, image_path, outputs, timings, peak_rss_gb=None, extra=None):
    """Write <output>.receipt.json and return its path.

    outputs: dict from export_mesh (paths + counts). File paths in it are
    hashed; non-path values are copied through.
    """
    files = {}
    for k, v in outputs.items():
        if isinstance(v, str) and os.path.isfile(v):
            files[k] = {"path": v, "sha256": sha256(v), "bytes": os.path.getsize(v)}
    image = None
    if image_path and os.path.isfile(image_path):
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                size = list(im.size)
        except Exception:
            size = None
        image = {"path": image_path, "sha256": sha256(image_path),
                 "bytes": os.path.getsize(image_path), "size": size}
    receipt = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "settings": settings,
        "input": image,
        "outputs": files,
        "mesh": {k: v for k, v in outputs.items() if isinstance(v, (int, float))},
        "bake": outputs.get("bake"),
        "timings_s": timings,
        "peak_rss_gb": peak_rss_gb,
        "env": _versions(),
    }
    if extra:
        receipt.update(extra)
    path = f"{output}.receipt.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    return path
