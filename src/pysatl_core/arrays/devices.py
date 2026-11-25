"""Devices supported by the common array interface."""

from enum import StrEnum


class Device(StrEnum):
    """Execution device selected for the process."""

    CPU = "cpu"


__all__ = ["Device"]
