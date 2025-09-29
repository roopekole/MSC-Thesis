"""
Comprehensive stock data management system.

This module provides intelligent data fetching, caching, and management
for stock market data with support for research environments including CSC.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import hashlib
import json
import time
import random
from functools import wraps

from .data_structures import StockData
from .config import ExperimentConfig


def rate_limit_retry(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator to handle rate limiting with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds before retrying
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"Rate limited. Waiting {delay:.2f} seconds before retry {attempt + 2}/{max_retries}")
                            time.sleep(delay)
                            continue
                    raise e
            return None
        return wrapper
    return decorator


class StockDataManager:
    """
    Comprehensive data management system for stock data with caching, 
    intelligent fetching, and portable storage for research environments.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the data manager with configuration.
        
        Args:
            config: ExperimentConfig object with all parameters
        """
        self.config = config
        
        # Setup directory structure
        project_root = Path(__file__).parent.parent
        self.storage_dir = project_root / config.data_storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cache_dir = self.storage_dir / "cache"
        self.metadata_dir = self.storage_dir / "metadata"
        self.backup_dir = self.storage_dir / "backup" 
        
        for dir_path in [self.cache_dir, self.metadata_dir, self.backup_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.metadata_dir / "data_registry.json"
        self.metadata = self._load_metadata()
        
        if config.enable_debug_output:
            print(f"Stock Data Manager initialized:")
            print(f"  Storage directory: {self.storage_dir}")
            print(f"  Cache format: {config.data_storage_format}")
            print(f"  Cache expiry: {config.cache_expiry_days} days")
            print(f"  Caching enabled: {config.enable_data_caching}")
    
    def _load_metadata(self) -> Dict:
        """Load data registry metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.config.enable_debug_output:
                    print(f"Warning: Could not load metadata file: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save data registry metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            if self.config.enable_debug_output:
                print(f"Warning: Could not save metadata: {e}")
    
    def _get_cache_key(self, symbol: str, start_date: str = None, end_date: str = None, 
                      period: str = None) -> str:
        """Generate unique cache key for data request."""
        key_data = {
            'symbol': symbol.upper(),
            'start_date': start_date,
            'end_date': end_date,
            'period': period
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached data."""
        if self.config.data_storage_format == "parquet":
            return self.cache_dir / f"{cache_key}.parquet"
        elif self.config.data_storage_format == "pickle":
            return self.cache_dir / f"{cache_key}.pkl"
        else:  # csv
            return self.cache_dir / f"{cache_key}.csv"
    
    def _is_data_fresh(self, cache_key: str) -> bool:
        """Check if cached data is still fresh."""
        if cache_key not in self.metadata:
            return False
        
        cached_date = datetime.fromisoformat(self.metadata[cache_key]['cached_at'])
        expiry_date = cached_date + timedelta(days=self.config.cache_expiry_days)
        
        return datetime.now() < expiry_date and not self.config.force_refresh_data
    
    def _save_data_to_cache(self, data: pd.DataFrame, cache_key: str, symbol: str, 
                           request_params: Dict):
        """Save data to cache with metadata."""
        if not self.config.enable_data_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save data in specified format
            if self.config.data_storage_format == "parquet":
                if self.config.data_compression:
                    data.to_parquet(cache_path, compression='gzip')
                else:
                    data.to_parquet(cache_path)
            elif self.config.data_storage_format == "pickle":
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:  # csv
                data.to_csv(cache_path)
            
            # Update metadata
            self.metadata[cache_key] = {
                'symbol': symbol.upper(),
                'cached_at': datetime.now().isoformat(),
                'file_path': str(cache_path),
                'data_rows': len(data),
                'date_range': {
                    'start': data.index.min().isoformat() if not data.empty else None,
                    'end': data.index.max().isoformat() if not data.empty else None
                },
                'request_params': request_params,
                'file_size_mb': cache_path.stat().st_size / (1024*1024) if cache_path.exists() else 0
            }
            
            self._save_metadata()
            
        except Exception as e:
            if self.config.enable_debug_output:
                print(f"Warning: Failed to cache data for {symbol}: {e}")
    
    def _load_data_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        if not self.config.enable_data_caching:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            if self.config.data_storage_format == "parquet":
                return pd.read_parquet(cache_path)
            elif self.config.data_storage_format == "pickle":
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            else:  # csv
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        except Exception as e:
            if self.config.enable_debug_output:
                print(f"Warning: Failed to load cached data: {e}")
            return None
    
    @rate_limit_retry(max_retries=3, base_delay=2.0)
    def _fetch_from_yfinance(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch data from yfinance with rate limiting."""
        time.sleep(random.uniform(0.5, 1.5))
        
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(interval="1d", **kwargs)
        
        if hist_data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Keep only OHLC columns and clean data
        ohlc_data = hist_data[["Open", "High", "Low", "Close"]].copy()
        
        # Convert timezone-aware index to timezone-naive for consistency
        if ohlc_data.index.tz is not None:
            ohlc_data.index = ohlc_data.index.tz_convert(None)
        
        return ohlc_data
    
    def get_stock_data(self, symbol: str, period: str = "6mo", 
                      start_date: str = None, end_date: str = None) -> StockData:
        """
        Get stock data with intelligent caching.
        
        Args:
            symbol: Stock symbol
            period: Time period (e.g., '1mo', '3mo', '6mo', '1y')
            start_date: Start date (YYYY-MM-DD) - overrides period
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            StockData object with OHLC data
        """
        symbol = symbol.upper()
        
        # Generate cache key
        cache_key = self._get_cache_key(symbol, start_date, end_date, period)
        
        # Check cache first
        if self.config.enable_data_caching and self._is_data_fresh(cache_key):
            cached_data = self._load_data_from_cache(cache_key)
            if cached_data is not None:
                if self.config.enable_debug_output:
                    print(f"Loaded {symbol} from cache ({len(cached_data)} days)")
                return StockData(symbol=symbol, ohlc=cached_data)
        
        # Fetch from API
        if self.config.enable_debug_output:
            print(f"Fetching {symbol} from Yahoo Finance...")
        
        try:
            # Prepare yfinance parameters
            yf_params = {}
            if start_date and end_date:
                yf_params['start'] = start_date
                yf_params['end'] = end_date
            elif end_date:
                yf_params['period'] = period
                yf_params['end'] = end_date
            else:
                yf_params['period'] = period
            
            ohlc_data = self._fetch_from_yfinance(symbol, **yf_params)
            
            # Save to cache
            request_params = {
                'period': period,
                'start_date': start_date,
                'end_date': end_date,
                'fetched_at': datetime.now().isoformat()
            }
            
            self._save_data_to_cache(ohlc_data, cache_key, symbol, request_params)
            
            if self.config.enable_debug_output:
                print(f"Fetched and cached {symbol} ({len(ohlc_data)} days)")
            return StockData(symbol=symbol, ohlc=ohlc_data)
        
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            raise
    
    def get_backtesting_data(self, symbol: str, cutoff_date: str, 
                           evaluation_days: int = 30) -> Dict:
        """
        Get data split for backtesting with intelligent caching.
        
        Args:
            symbol: Stock symbol
            cutoff_date: Knowledge cutoff date (YYYY-MM-DD)
            evaluation_days: Days after cutoff for evaluation
        
        Returns:
            Dictionary with 'historical_data', 'evaluation_data', etc.
        """
        cutoff_datetime = pd.to_datetime(cutoff_date)
        start_date = (cutoff_datetime - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = (cutoff_datetime + pd.Timedelta(days=evaluation_days + 10)).strftime('%Y-%m-%d')
        
        # Get full dataset
        full_data = self.get_stock_data(symbol, start_date=start_date, end_date=end_date)
        
        # Split data
        cutoff_datetime = cutoff_datetime.tz_localize(None) if cutoff_datetime.tz else cutoff_datetime
        
        training_mask = full_data.ohlc.index <= cutoff_datetime
        evaluation_mask = full_data.ohlc.index > cutoff_datetime
        
        training_data = full_data.ohlc[training_mask].copy()
        evaluation_data = full_data.ohlc[evaluation_mask].copy()
        
        if training_data.empty or evaluation_data.empty:
            raise ValueError(f"Insufficient data for backtesting {symbol}")
        
        # Get prediction context (last 15 days before cutoff)
        prediction_context = training_data.tail(15).copy()
        
        return {
            'symbol': symbol,
            'historical_data': StockData(symbol=symbol, ohlc=training_data),
            'prediction_context': StockData(symbol=symbol, ohlc=prediction_context),
            'prediction_date': cutoff_datetime,
            'evaluation_data': StockData(symbol=symbol, ohlc=evaluation_data),
            'total_historical_days': len(training_data),
            'evaluation_days': len(evaluation_data)
        }
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "6mo", 
                           delay_between: float = 1.0) -> List[StockData]:
        """
        Get multiple stocks with intelligent batching and caching.
        
        Args:
            symbols: List of stock symbols
            period: Time period for each stock
            delay_between: Delay between API calls (for non-cached data)
        
        Returns:
            List of StockData objects
        """
        results = []
        symbols_to_fetch = []
        cached_count = 0
        
        if self.config.enable_debug_output:
            print(f"Processing {len(symbols)} stocks...")
        
        # First pass: check what's in cache
        for symbol in symbols:
            cache_key = self._get_cache_key(symbol.upper(), period=period)
            
            if self.config.enable_data_caching and self._is_data_fresh(cache_key):
                try:
                    cached_data = self._load_data_from_cache(cache_key)
                    if cached_data is not None:
                        results.append(StockData(symbol=symbol.upper(), ohlc=cached_data))
                        cached_count += 1
                        continue
                except Exception:
                    pass
            
            symbols_to_fetch.append(symbol)
        
        if self.config.enable_debug_output:
            print(f"Loaded {cached_count} stocks from cache")
        
        # Second pass: fetch missing data
        if symbols_to_fetch:
            if self.config.enable_debug_output:
                print(f"Fetching {len(symbols_to_fetch)} stocks from API...")
            
            for i, symbol in enumerate(symbols_to_fetch):
                try:
                    stock_data = self.get_stock_data(symbol, period=period)
                    results.append(stock_data)
                    
                    # Add delay between API requests (not for cached data)
                    if i < len(symbols_to_fetch) - 1:
                        time.sleep(delay_between)
                
                except Exception as e:
                    print(f"Failed to fetch {symbol}: {e}")
        
        if self.config.enable_debug_output:
            print(f"Successfully loaded {len(results)} stocks total")
        return results
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        total_files = len(self.metadata)
        total_size_mb = sum(entry.get('file_size_mb', 0) for entry in self.metadata.values())
        
        fresh_data = sum(1 for key in self.metadata.keys() if self._is_data_fresh(key))
        stale_data = total_files - fresh_data
        
        return {
            'total_cached_datasets': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'fresh_datasets': fresh_data,
            'stale_datasets': stale_data,
            'cache_directory': str(self.cache_dir),
            'storage_format': self.config.data_storage_format
        }
    
    def clear_stale_cache(self):
        """Remove stale cached data."""
        removed_count = 0
        
        for cache_key in list(self.metadata.keys()):
            if not self._is_data_fresh(cache_key):
                try:
                    cache_path = self._get_cache_path(cache_key)
                    if cache_path.exists():
                        cache_path.unlink()
                    
                    del self.metadata[cache_key]
                    removed_count += 1
                
                except Exception as e:
                    if self.config.enable_debug_output:
                        print(f"Warning: Failed to remove stale cache {cache_key}: {e}")
        
        if removed_count > 0:
            self._save_metadata()
            if self.config.enable_debug_output:
                print(f"Removed {removed_count} stale cache files")
        else:
            if self.config.enable_debug_output:
                print("No stale cache files to remove")
    
    def create_data_backup(self) -> str:
        """Create a portable backup of all cached data."""
        backup_file = self.backup_dir / f"stock_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        
        try:
            import tarfile
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(self.cache_dir, arcname='cache')
                tar.add(self.metadata_file, arcname='metadata/data_registry.json')
            
            if self.config.enable_debug_output:
                print(f"Created data backup: {backup_file}")
                print(f"Size: {backup_file.stat().st_size / (1024*1024):.1f} MB")
            
            return str(backup_file)
        
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None


def create_sample_data(symbol: str = "NVDA") -> StockData:
    """
    Create sample OHLC data for testing purposes.
    
    Args:
        symbol: Stock symbol to use for sample data
        
    Returns:
        StockData object with synthetic OHLC data
    """
    dates = pd.date_range(start="2024-06-01", end="2024-09-29", freq="D")
    base_price = 150.0
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    ohlc_df = pd.DataFrame({
        "Open": [p * random.uniform(0.98, 1.02) for p in prices],
        "High": [p * random.uniform(1.00, 1.05) for p in prices],
        "Low": [p * random.uniform(0.95, 1.00) for p in prices],
        "Close": prices
    }, index=dates)
    
    return StockData(symbol=symbol.upper(), ohlc=ohlc_df)
