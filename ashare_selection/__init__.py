"""A-share candidate-pool research pipeline."""

from .config import AppConfig, load_config
from .generated_features import FeatureDefinition
from .pipeline import CandidateSelector
from .tushare_source import TushareDataSource

__all__ = [
    "AppConfig",
    "CandidateSelector",
    "FeatureDefinition",
    "TushareDataSource",
    "load_config",
]
__version__ = "0.2.0"
