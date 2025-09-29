"""
Utility functions for data management and operations.

This module provides high-level utility functions for common data operations
including bulk fetching, backtesting preparation, and CSC deployment.
"""

from typing import List, Tuple
from pathlib import Path
import time

from .data_manager import StockDataManager
from .config import ExperimentConfig
from .data_structures import StockData


def demonstrate_data_management(data_manager: StockDataManager):
    """
    Demonstrate the data management system capabilities.
    
    Args:
        data_manager: Initialized StockDataManager instance
    """
    print("Data Management System Demonstration")
    print("=" * 50)
    
    # Show current cache status
    cache_info = data_manager.get_cache_info()
    print(f"\nCurrent Cache Status:")
    print(f"   Storage Directory: {cache_info['cache_directory']}")
    print(f"   Cached Datasets: {cache_info['total_cached_datasets']}")
    print(f"   Fresh Data: {cache_info['fresh_datasets']}")
    print(f"   Stale Data: {cache_info['stale_datasets']}")
    print(f"   Total Size: {cache_info['total_size_mb']} MB")
    print(f"   Format: {cache_info['storage_format']}")
    
    # Demonstrate fetching a single stock
    print(f"\nTesting Single Stock Fetch:")
    try:
        test_symbol = "AAPL"
        stock_data = data_manager.get_stock_data(test_symbol, period="1mo")
        print(f"   Successfully loaded {test_symbol}: {len(stock_data.ohlc)} days")
        print(f"   Date range: {stock_data.ohlc.index.min().date()} to {stock_data.ohlc.index.max().date()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Show cache efficiency
    print(f"\nCache Efficiency:")
    if cache_info['total_cached_datasets'] > 0:
        efficiency = (cache_info['fresh_datasets'] / cache_info['total_cached_datasets']) * 100
        print(f"   Fresh Data Ratio: {efficiency:.1f}%")
        avg_size = cache_info['total_size_mb'] / cache_info['total_cached_datasets']
        print(f"   Average Dataset Size: {avg_size:.2f} MB")
    else:
        print(f"   No cached data available yet")
    
    print(f"\nReady for efficient data operations!")


def bulk_fetch_config_stocks(data_manager: StockDataManager, 
                           config: ExperimentConfig) -> List[StockData]:
    """
    Bulk fetch all stocks from configuration for caching.
    
    Args:
        data_manager: Initialized StockDataManager instance
        config: Experiment configuration
        
    Returns:
        List of successfully fetched StockData objects
    """
    print(f"Bulk Fetching Stocks from Configuration")
    print(f"=" * 50)
    
    symbols = config.test_symbols
    print(f"Processing {len(symbols)} symbols from configuration...")
    
    try:
        results = data_manager.get_multiple_stocks(
            symbols=symbols,
            period=config.data_period,
            delay_between=1.0  # Conservative rate limiting
        )
        
        successful = len(results)
        failed = len(symbols) - successful
        
        print(f"\nBulk Fetch Results:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {(successful/len(symbols)*100):.1f}%")
        
        if successful > 0:
            total_days = sum(len(stock.ohlc) for stock in results)
            avg_days = total_days / successful
            print(f"   Average Days per Stock: {avg_days:.1f}")
        
        return results
        
    except Exception as e:
        print(f"Bulk fetch failed: {e}")
        return []


def prepare_backtesting_dataset(data_manager: StockDataManager, 
                              config: ExperimentConfig) -> Tuple[int, List[str]]:
    """
    Pre-fetch and cache backtesting data for all configured symbols.
    
    Args:
        data_manager: Initialized StockDataManager instance
        config: Experiment configuration
        
    Returns:
        Tuple of (successful_count, failed_symbols)
    """
    print(f"Preparing Backtesting Dataset")
    print(f"=" * 50)
    
    symbols = config.test_symbols
    cutoff_date = config.prediction_cutoff_date
    evaluation_days = config.evaluation_days
    
    print(f"Symbols: {len(symbols)}")
    print(f"Cutoff Date: {cutoff_date}")
    print(f"Evaluation Period: {evaluation_days} days")
    
    successful_fetches = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols):
        try:
            if config.enable_debug_output:
                print(f"   Fetching {symbol} ({i+1}/{len(symbols)})...")
            
            # This will cache the full dataset needed for backtesting
            training_data, evaluation_data = data_manager.get_backtesting_data(
                symbol, cutoff_date, evaluation_days
            )
            
            successful_fetches += 1
            
            # Small delay to be respectful to API
            if i < len(symbols) - 1:
                time.sleep(0.5)
            
        except Exception as e:
            print(f"   Failed {symbol}: {e}")
            failed_symbols.append(symbol)
    
    print(f"\nBacktesting Dataset Preparation:")
    print(f"   Successful: {successful_fetches}")
    print(f"   Failed: {len(failed_symbols)}")
    print(f"   Success Rate: {(successful_fetches/len(symbols)*100):.1f}%")
    
    if failed_symbols:
        print(f"   Failed Symbols: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")
    
    # Update cache info
    cache_info = data_manager.get_cache_info()
    print(f"   Total Cache Size: {cache_info['total_size_mb']} MB")
    
    return successful_fetches, failed_symbols


def optimize_data_storage(data_manager: StockDataManager):
    """
    Optimize data storage by cleaning stale cache and creating backup.
    
    Args:
        data_manager: Initialized StockDataManager instance
    """
    print(f"Optimizing Data Storage")
    print(f"=" * 50)
    
    # Show current status
    cache_info = data_manager.get_cache_info()
    print(f"Before Optimization:")
    print(f"   Total Datasets: {cache_info['total_cached_datasets']}")
    print(f"   Stale Datasets: {cache_info['stale_datasets']}")
    print(f"   Total Size: {cache_info['total_size_mb']} MB")
    
    # Clean stale cache
    if cache_info['stale_datasets'] > 0:
        print(f"\nCleaning Stale Cache...")
        data_manager.clear_stale_cache()
    else:
        print(f"\nNo stale data to clean")
    
    # Create backup if we have significant data
    if cache_info['total_size_mb'] > 1.0:
        print(f"\nCreating Data Backup...")
        backup_file = data_manager.create_data_backup()
        if backup_file:
            print(f"   Backup created: {Path(backup_file).name}")
    
    # Show final status
    final_cache_info = data_manager.get_cache_info()
    print(f"\nAfter Optimization:")
    print(f"   Total Datasets: {final_cache_info['total_cached_datasets']}")
    print(f"   Fresh Datasets: {final_cache_info['fresh_datasets']}")
    print(f"   Total Size: {final_cache_info['total_size_mb']} MB")
    
    print(f"\nData storage optimization complete!")


def export_for_csc(data_manager: StockDataManager, config: ExperimentConfig) -> Tuple[str, str]:
    """
    Create a portable data package for CSC (or other compute environments).
    
    Args:
        data_manager: Initialized StockDataManager instance
        config: Experiment configuration
        
    Returns:
        Tuple of (backup_file_path, instructions_file_path) or (None, None) if failed
    """
    print(f"Creating Portable Data Package for CSC")
    print(f"=" * 50)
    
    try:
        # Create backup
        backup_file = data_manager.create_data_backup()
        
        if backup_file:
            backup_path = Path(backup_file)
            
            # Create transfer instructions
            instructions_file = backup_path.parent / "CSC_TRANSFER_INSTRUCTIONS.txt"
            
            with open(instructions_file, 'w') as f:
                f.write("CSC Data Transfer Instructions\n")
                f.write("="*40 + "\n\n")
                f.write(f"1. Upload this backup file to CSC: {backup_path.name}\n")
                f.write("2. Extract the backup:\n")
                f.write(f"   tar -xzf {backup_path.name}\n\n")
                f.write("3. Set up the notebook environment:\n")
                f.write("   - Ensure data_storage_dir points to extracted cache directory\n")
                f.write("   - Set enable_data_caching = True in CONFIG\n")
                f.write("   - The system will automatically use cached data\n\n")
                f.write("4. Configuration for CSC:\n")
                f.write("   CONFIG.data_storage_dir = 'storage/stock_data_cache'\n")
                f.write("   CONFIG.enable_data_caching = True\n")
                f.write("   CONFIG.cache_expiry_days = 30  # Longer for CSC\n")
                f.write("   CONFIG.enable_debug_output = False  # Reduce output\n\n")
                f.write("5. All your cached stock data will be available immediately!\n")
            
            print(f"Backup Package: {backup_path.name}")
            print(f"Instructions: {instructions_file.name}")
            print(f"Package Size: {backup_path.stat().st_size / (1024*1024):.1f} MB")
            
            cache_info = data_manager.get_cache_info()
            print(f"Contains {cache_info['total_cached_datasets']} datasets")
            
            print(f"\nReady for CSC transfer!")
            print(f"Upload both files: {backup_path.name} and {instructions_file.name}")
            
            return str(backup_path), str(instructions_file)
        
    except Exception as e:
        print(f"Export failed: {e}")
        return None, None
