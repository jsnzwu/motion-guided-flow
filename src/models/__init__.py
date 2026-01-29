from wickit.models import MODELS

from .mfrrnet.mfrrnet import MFRRNetModel

MODELS.register_module(name="MFRRNetModel")(MFRRNetModel)

__all__ = [
    "MFRRNetModel",
]
