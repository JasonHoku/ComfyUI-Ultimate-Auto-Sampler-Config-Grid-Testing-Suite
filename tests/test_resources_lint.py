"""Lint for the dashboard resources: leftover merge markers silently break CSS/JS."""
import os
import re

_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
_MARKER = re.compile(r"^(<<<<<<<|=======|>>>>>>>)( |$)")


def test_no_merge_markers_in_resources():
    bad = []
    for name in sorted(os.listdir(_RES)):
        path = os.path.join(_RES, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                if _MARKER.match(line.rstrip("\r\n")):
                    bad.append(f"{name}:{n}: {line.strip()}")
    assert not bad, "merge markers left in resources: " + "; ".join(bad)
