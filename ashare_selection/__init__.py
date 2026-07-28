"""A-share candidate-pool research pipeline."""

from .config import AppConfig, load_config
from .pipeline import CandidateSelector
from .tushare_source import TushareDataSource

__all__ = [
    "AppConfig",
    "CandidateSelector",
    "TushareDataSource",
    "load_config",
]
__version__ = "0.1.0"
