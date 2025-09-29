"""
Data processing package for MSc thesis project.

This package provides data fetching, caching, and management functionality
for stock market data used in zero-shot forecasting experiments.
"""

from .data_structures import StockData, TradingSignal, SignalType, ModelProvider, ModelConfig
from .data_manager import StockDataManager, create_sample_data
from .config import ExperimentConfig
from . import utils

__all__ = [
    'StockData',
    'TradingSignal', 
    'SignalType',
    'ModelProvider',
    'ModelConfig',
    'StockDataManager',
    'ExperimentConfig',
    'create_sample_data',
    'utils'
]
