"""InfluencerFormer: MaskFormer with Influencer Loss replacing Hungarian matching."""

__version__ = "0.2.0"

from .losses import MaskInfluencerLoss
from .models import InfluencerCriterion
