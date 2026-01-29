from wickit.runner import RUNNERS

from .mfrrnet_runner import MFRRNetRunner

RUNNERS.register_module(name="MFRRNetRunner")(MFRRNetRunner)

__all__ = [
    "MFRRNetRunner",
]
