"""
Configuration management for MSc thesis experiments.

This module provides centralized configuration for all experimental parameters,
data management settings, and model configurations.
"""

from dataclasses import dataclass
from typing import List
from .data_structures import ModelProvider


@dataclass
class ExperimentConfig:
    """
    Centralized configuration for all experiment parameters.
    
    This class contains all configurable parameters for the MSc thesis experiments,
    organized by functional area for easy management and reproducibility.
    """
    
    # Experiment Identification (keep minimal for data management context)
    experiment_name: str = "zero_shot_forecasting_v1"
    
    # Data Management Parameters
    enable_data_caching: bool = True
    data_storage_dir: str = "storage/stock_data_cache"
    data_storage_format: str = "parquet"  # Options: 'parquet', 'pickle', 'csv'
    cache_expiry_days: int = 7  # Days before data is considered stale
    force_refresh_data: bool = False  # Force re-download all data
    data_compression: bool = True  # Compress stored data files
    enable_debug_output: bool = True  # Enable debug output for development
    
    # Data Parameters
    test_symbols: List[str] = None  # Will be set in __post_init__
    data_period: str = "6mo"  # Extended for backtesting
    fallback_symbol: str = "IONQ"  # For sample data when rate limited
    
    # Backtesting Parameters
    backtest_enabled: bool = True
    prediction_cutoff_date: str = "2024-06-01"  # Use data up to this date for predictions
    evaluation_days: int = 30  # Evaluate performance over next 30 days
    
    # Model Parameters
    model_name: str = "gpt-4o-mini"  # Azure OpenAI model name
    temperature: float = 1.2  # Temperature for response generation
    top_p: float = 0.9  # Top-p sampling parameter
    max_tokens: int = 100  # Maximum tokens in model response
    n_completions: int = 5  # Number of completions for ensemble forecasting
    min_completion_rate: float = 0.4  # Minimum fraction of successful completions required
    
    # OHLC Analysis Parameters
    use_ohlc_analysis: bool = True  # Enable enhanced OHLC volatility analysis
    ohlc_history_days: int = 15  # Number of days of OHLC history to use for analysis
    forecast_days: int = 1  # Number of days to forecast ahead
    
    # Signal Generation Parameters
    base_confidence: float = 0.5  # Base confidence score
    strong_threshold: float = 0.025  # 2.5% threshold for strong signals
    moderate_threshold: float = 0.01  # 1.0% threshold for moderate signals
    weak_threshold: float = 0.005  # 0.5% threshold for weak signals
    spread_confidence_weight: float = 0.3  # Weight for spread-based confidence boost
    consensus_confidence_weight: float = 0.2  # Weight for consensus-based confidence boost
    magnitude_confidence_weight: float = 0.2  # Weight for magnitude-based confidence boost
    
    # Experiment Logging
    enable_experiment_logging: bool = True  # Enable comprehensive experiment logging
    
    # Visual Analysis Parameters  
    enable_visual_analysis: bool = True  # Master switch for all visual analysis
    enable_performance_plots: bool = True  # Performance and confidence visualizations
    enable_sector_analysis: bool = True  # Sector-wise performance breakdown
    enable_signal_distribution: bool = True  # Signal type and confidence distribution
    enable_time_series_plots: bool = True  # Individual stock price and prediction plots
    visual_output_dpi: int = 150  # High quality for academic publication
    

    
    def __post_init__(self):
        """Set default test symbols if not provided."""
        if self.test_symbols is None:
            self.test_symbols = [
                # Mega Cap Technology
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE", "CRM",
                
                # Technology & Software
                "AMD", "INTC", "ORCL", "NOW", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX",
                
                # Finance & Banking
                "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC",
                
                # Healthcare & Pharmaceuticals  
                "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
                
                # Consumer & Retail
                "WMT", "PG", "KO", "PEP", "NKE", "MCD", "SBUX", "TGT", "HD", "LOW",
                
                # Energy & Utilities
                "XOM", "CVX", "COP", "EOG", "SLB", "NEE", "DUK", "SO", "AEP", "EXC",
                
                # Industrial & Manufacturing
                "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "EMR",
                
                # Telecommunications & Media
                "VZ", "T", "TMUS", "DIS", "CMCSA", "CHTR", "VIA", "PARA", "WBD", "FOXA",
                
                # Real Estate & REITs
                "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "UDR", "CPT",
                
                # Materials & Chemicals
                "LIN", "APD", "ECL", "SHW", "DD", "DOW", "FCX", "NUE", "STLD", "VMC"
            ]
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        from pathlib import Path
        
        config_dict = self.to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def get_csc_config(self) -> 'ExperimentConfig':
        """Get CSC-optimized configuration."""
        csc_config = ExperimentConfig()
        
        # Copy all current settings
        for field in self.__dataclass_fields__:
            setattr(csc_config, field, getattr(self, field))
        
        # CSC-specific optimizations
        csc_config.data_storage_dir = "storage/stock_data_cache"  # Relative path
        csc_config.cache_expiry_days = 30  # Longer cache for compute clusters
        csc_config.enable_debug_output = False  # Reduce output for batch processing
        csc_config.data_compression = True  # Essential for storage efficiency
        
        return csc_config
