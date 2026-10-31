import json

from PIL import Image

from trellis3.receipt import sha256, write_receipt


def test_receipt_contents(tmp_path):
    img = tmp_path / "in.png"
    Image.new("RGB", (7, 5)).save(img)
    glb = tmp_path / "o.glb"
    glb.write_bytes(b"glb-bytes")
    outputs = {"glb": str(glb), "obj": str(tmp_path / "missing.obj"), "faces": 12, "bake": "metal"}
    path = write_receipt(str(tmp_path / "o"), {"seed": 3, "pipeline_type": "512"}, str(img),
                         outputs, {"generate": 1.5, "bake": 0.5}, peak_rss_gb=13.2,
                         extra={"job_id": "x1"})
    r = json.loads(open(path).read())
    assert path.endswith("o.receipt.json")
    assert r["settings"]["seed"] == 3
    assert r["input"]["size"] == [7, 5] and r["input"]["sha256"] == sha256(str(img))
    assert r["outputs"]["glb"]["sha256"] == sha256(str(glb)) and r["outputs"]["glb"]["bytes"] == 9
    assert "obj" not in r["outputs"]  # missing files are skipped, not fatal
    assert r["mesh"]["faces"] == 12 and r["bake"] == "metal"
    assert r["timings_s"]["generate"] == 1.5 and r["peak_rss_gb"] == 13.2
    assert r["job_id"] == "x1" and "python" in r["env"]


def test_receipt_without_image(tmp_path):
    path = write_receipt(str(tmp_path / "n"), {}, None, {}, {})
    assert json.loads(open(path).read())["input"] is None
