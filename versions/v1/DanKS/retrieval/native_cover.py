from __future__ import annotations

"""Public build: native acceleration is not distributed."""


def available() -> bool:
    return False


def module():
    raise RuntimeError("native acceleration is not available in the public build")
