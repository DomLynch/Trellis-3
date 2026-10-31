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
}

# Values a user could have typed explicitly; anything else means "use preset".
_DEFAULTS = {"pipeline_type": None, "steps": None, "texture_size": None, "no_texture": False}


def apply_preset(name, pipeline_type=None, steps=None, texture_size=None, no_texture=False):
    """Merge preset values with explicit overrides. Returns a settings dict.

    With no preset the upstream defaults are used (512, 12 steps, 1024 tex).
    """
    if name is None:
        base = {"pipeline_type": "512", "steps": None, "texture_size": 1024, "no_texture": False}
    elif name in PRESETS:
        base = dict(PRESETS[name])
    else:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")

    explicit = {"pipeline_type": pipeline_type, "steps": steps,
                "texture_size": texture_size, "no_texture": no_texture}
    for k, v in explicit.items():
        if v != _DEFAULTS[k]:
            base[k] = v
    return base
