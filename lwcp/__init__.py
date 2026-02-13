"""Leverage-Weighted Conformal Prediction."""

from .conformal import LWCP
from .diagnostics import WeightAlignmentResult, diagnose_weight_alignment
from .leverage import FeatureSpaceLeverageComputer, LeverageComputer, mlp_feature_extractor
from .weights import (
    ConstantWeight,
    InverseRootLeverageWeight,
    PowerLawWeight,
    WeightFunction,
    WeightSelector,
)

__version__ = "0.1.0"

__all__ = [
    "LWCP",
    "LeverageComputer",
    "FeatureSpaceLeverageComputer",
    "mlp_feature_extractor",
    "WeightFunction",
    "ConstantWeight",
    "InverseRootLeverageWeight",
    "PowerLawWeight",
    "WeightSelector",
    "diagnose_weight_alignment",
    "WeightAlignmentResult",
]
