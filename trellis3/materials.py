"""
Texture-derived materials pass for a baked GLB (CPU, numpy/cv2).

TRELLIS bakes one base-colour texture and a flat metallic/roughness. This writes a glTF metallicRoughness
texture (G = roughness, B = metallic) classified from the colour texture, so a turntable or the game shows
gloss on gold and brocade, a soft sheen on skin, sharper highlights on hair and matte cloth:

    gold / brocade trim   warm yellow-orange, saturated, bright   roughness 0.28, metallic 0.85
    hair / very dark      value < 0.18                            roughness 0.38 (tight specular), metallic 0
    skin                  warm low-saturation mid-bright          roughness 0.50, metallic 0
    everything else       cloth                                   roughness 0.75, metallic 0

    python -m trellis3.materials in.glb out.glb
"""
import json
import sys

import numpy as np

CLASSES = {"gold": (0.28, 0.85), "hair": (0.38, 0.0), "skin": (0.50, 0.0), "cloth": (0.75, 0.0)}


def classify(rgb):
    """rgb uint8 (H,W,3) -> dict of boolean masks (gold, hair, skin) ; cloth is the remainder."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    h, s, v = hsv[..., 0] * 360.0 / 255.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gold = (h >= 32) & (h <= 62) & (s > 0.35) & (v > 0.38)
    hair = (v < 0.18) & ~gold
    skin = (h <= 40) & (s > 0.08) & (s < 0.55) & (v > 0.35) & ~gold & ~hair
    return {"gold": gold, "hair": hair, "skin": skin}


def metallic_roughness(rgb, blur=3):
    """glTF metallicRoughness image (R unused, G roughness, B metallic) from a colour texture."""
    import cv2
    masks = classify(rgb)
    rough = np.full(rgb.shape[:2], CLASSES["cloth"][0], np.float32)
    metal = np.zeros(rgb.shape[:2], np.float32)
    for name in ("skin", "hair", "gold"):           # later classes win
        r, m = CLASSES[name]
        rough[masks[name]] = r
        metal[masks[name]] = m
    k = blur * 2 + 1
    rough, metal = cv2.GaussianBlur(rough, (k, k), 0), cv2.GaussianBlur(metal, (k, k), 0)
    out = np.zeros(rgb.shape, np.uint8)
    out[..., 1] = np.clip(rough * 255, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(metal * 255, 0, 255).astype(np.uint8)
    return out, {k_: round(100 * float(v_.mean()), 1) for k_, v_ in masks.items()}


def apply_materials(glb_in, glb_out, log=print):
    import trimesh
    from PIL import Image
    sc = trimesh.load(glb_in)
    g = list(sc.geometry.values())[0] if hasattr(sc, "geometry") else sc
    m = g.visual.material
    rgb = np.asarray(m.baseColorTexture.convert("RGB"))
    mr, share = metallic_roughness(rgb)
    m.metallicRoughnessTexture = Image.fromarray(mr)
    m.metallicFactor = 1.0
    m.roughnessFactor = 1.0
    sc.export(glb_out)
    json.dump({"in": glb_in, "out": glb_out, "class_share_pct": share, "classes": CLASSES},
              open(glb_out + ".materials.json", "w"), indent=1)
    log(f"materials pass: {share} -> {glb_out}")
    return share


if __name__ == "__main__":
    apply_materials(sys.argv[1], sys.argv[2])
