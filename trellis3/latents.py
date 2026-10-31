"""
Save / load the sampling state produced by trellis3.stages.

A state file is a torch.save of a plain dict with CPU tensors. At 1024 it is
tens of MB (49K tokens x 32 channels x 2 latents, plus coordinates), so
writing it after every stage is free compared with the minutes the stage
took. Decoding from it needs only the two decoders, not the five flow models.
"""

import os
import time

import torch

FORMAT = 1


def _to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu")
    if isinstance(v, dict):
        return {k: _to_cpu(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_to_cpu(x) for x in v)
    return v


def save_state(path, state):
    """Atomically write the state. Returns the path."""
    payload = {"format": FORMAT, "saved_at": time.time(), "state": _to_cpu(state)}
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Trellis-3 latents file")
    return payload["state"]


def describe(state):
    done = state.get("done", [])
    parts = [f"pipeline={state.get('pipeline_type')}", f"seed={state.get('seed')}", f"done={done}"]
    if "coords" in state:
        parts.append(f"voxels={state['coords'].shape[0]:,}")
    if "shape_feats" in state:
        parts.append(f"tokens={state['shape_feats'].shape[0]:,}@{state.get('res')}")
    return " ".join(parts)
