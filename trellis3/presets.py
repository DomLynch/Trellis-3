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
    # Pixal3D (TencentARC) in its own venv; generate.py hands over to trellis3.pixal3d. Values here only keep the
    # CLI choices and receipts consistent, the Pixal3D pipeline sets its own resolution and bake.
    "pixal3d": {"pipeline_type": "1024_cascade", "steps": None, "texture_size": 4096, "no_texture": False,
                "bake_faces": 1_000_000, "normal_map": False, "two_stage": False},
}

_BASE = {"pipeline_type": "512", "steps": None, "texture_size": 1024, "no_texture": False,
         "bake_faces": 200_000, "normal_map": False, "two_stage": False, "max_num_tokens": None}

# Cascade token budget. Upstream caps at 49152 and silently drops the
# resolution when a shape needs more. With 48 GB+ of memory there is no
# reason to lose resolution, so the default doubles; small machines keep the
# upstream cap because tokens are what the attention memory scales with.
TOKENS_SMALL = 49152
TOKENS_LARGE = 98304
LARGE_MACHINE_GB = 48


# Runtime memory defaults of the hero preset (issue-188 work, measured on the 1536 head: decode 19.9 GiB, export 12.6 GiB RSS-side, vs 48 GB GPU / 49 GB host untiled).
# Other presets keep today's behaviour. An environment variable already set always wins (0 turns the feature off).
HERO_RUNTIME_DEFAULTS = {"TRELLIS3_TILE_VOXELS": "256", "TRELLIS3_GPU_SIMPLIFY": "1"}


def hero_runtime_defaults(preset, env):
    """Apply the hero defaults to `env` (os.environ) with setdefault and return the list of (name, value, source) for the log line; [] for any other preset."""
    if preset != "hero":
        return []
    out = []
    for k, v in HERO_RUNTIME_DEFAULTS.items():
        src = "environment" if k in env else "hero default"
        env.setdefault(k, v)
        out.append((k, env[k], src))
    return out


def default_max_tokens(total_gb=None):
    if total_gb is None:
        try:
            from trellis3.memory import total_memory_gb
            total_gb = total_memory_gb()
        except Exception:
            total_gb = None
    return TOKENS_LARGE if (total_gb or 0) >= LARGE_MACHINE_GB else TOKENS_SMALL

# Values a user could have typed explicitly; anything else means "use preset".
_DEFAULTS = {"pipeline_type": None, "steps": None, "texture_size": None, "no_texture": False,
             "bake_faces": None, "normal_map": False, "two_stage": False, "max_num_tokens": None}


def apply_preset(name, pipeline_type=None, steps=None, texture_size=None, no_texture=False,
                 bake_faces=None, normal_map=False, two_stage=False, single_process=False,
                 max_num_tokens=None):
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
                "two_stage": two_stage, "max_num_tokens": max_num_tokens}
    for k, v in explicit.items():
        if v != _DEFAULTS[k]:
            base[k] = v
    if single_process:
        base["two_stage"] = False
    if base["max_num_tokens"] is None:
        base["max_num_tokens"] = default_max_tokens()
    return base
