"""
Draft / final presets for generate.py and serve.py.

`draft` is for the 4-of-5 runs you throw away: lowest resolution, fewer
sampler steps, no texture bake. On an M4 Pro that is roughly a minute of
generation instead of three to five. `final` is the upstream default quality.
Explicit flags always override the preset.
"""

PRESETS = {
    "draft": {"pipeline_type": "512", "steps": 8, "texture_size": 512, "no_texture": True},
    "final": {"pipeline_type": "1024_cascade", "steps": None, "texture_size": 2048, "no_texture": False},
    # Desktop master: everything the phone version is later cut down from.
    "master": {"pipeline_type": "1024_cascade", "steps": None, "texture_size": 4096, "no_texture": False,
               "bake_faces": 1_000_000, "normal_map": True, "two_stage": True},
    # Hero faces on a 64 GB machine: 1536 shape, 8K textures, 2M-face bake.
    "hero": {"pipeline_type": "1536_cascade", "steps": None, "texture_size": 8192, "no_texture": False,
             "bake_faces": 2_000_000, "normal_map": True, "two_stage": True},
}

_BASE = {"pipeline_type": "512", "steps": None, "texture_size": 1024, "no_texture": False,
         "bake_faces": 200_000, "normal_map": False, "two_stage": False}

# Values a user could have typed explicitly; anything else means "use preset".
_DEFAULTS = {"pipeline_type": None, "steps": None, "texture_size": None, "no_texture": False,
             "bake_faces": None, "normal_map": False, "two_stage": False}


def apply_preset(name, pipeline_type=None, steps=None, texture_size=None, no_texture=False,
                 bake_faces=None, normal_map=False, two_stage=False, single_process=False):
    """Merge preset values with explicit overrides. Returns a settings dict.

    With no preset the upstream defaults are used (512, 12 steps, 1024 tex,
    200K-face bake, no normal map, single process).
    """
    if name is None:
        base = dict(_BASE)
    elif name in PRESETS:
        base = {**_BASE, **PRESETS[name]}
    else:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")

    explicit = {"pipeline_type": pipeline_type, "steps": steps, "texture_size": texture_size,
                "no_texture": no_texture, "bake_faces": bake_faces, "normal_map": normal_map,
                "two_stage": two_stage}
    for k, v in explicit.items():
        if v != _DEFAULTS[k]:
            base[k] = v
    if single_process:
        base["two_stage"] = False
    return base
