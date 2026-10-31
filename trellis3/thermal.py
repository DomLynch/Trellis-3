"""
Thermal guard for fanless Macs.

trellis-mac's README measured the same job going from ~3.5 min to ~36 min
once an M4 Pro had been hot for an hour. A MacBook Air throttles sooner.
`pmset -g therm` reports `CPU_Speed_Limit` (100 = no throttling). Before a
job we can wait for it to recover instead of burning ten times the time.

Everything here degrades gracefully: if pmset is missing or unparsable the
guard reports "unknown" and never blocks.
"""

import re
import subprocess
import time

_LIMIT_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")


def parse_speed_limit(text):
    """Extract CPU_Speed_Limit (0-100) from `pmset -g therm` output, or None."""
    m = _LIMIT_RE.search(text or "")
    return int(m.group(1)) if m else None


def cpu_speed_limit():
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_speed_limit(out)


def wait_until_cool(min_speed=85, max_wait=600, poll=15, log=print):
    """Block until CPU_Speed_Limit >= min_speed or max_wait seconds pass.

    Returns the last observed limit (None if unavailable).
    """
    deadline = time.time() + max_wait
    limit = cpu_speed_limit()
    if limit is None or limit >= min_speed:
        return limit
    log(f"Thermal guard: CPU_Speed_Limit={limit} (< {min_speed}); waiting to cool...")
    while time.time() < deadline:
        time.sleep(poll)
        limit = cpu_speed_limit()
        if limit is None or limit >= min_speed:
            break
    log(f"Thermal guard: continuing at CPU_Speed_Limit={limit}")
    return limit
