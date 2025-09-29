#!/usr/bin/env python3
"""
Data management CLI script for MSc thesis project.

This script provides command-line interface for common data management operations
including bulk fetching, cache management, and CSC deployment preparation.

Usage:
    python data_operations.py --help
    python data_operations.py fetch-all
    python data_operations.py prepare-backtesting
    python data_operations.py export-csc
    python data_operations.py cache-info
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_processing import ExperimentConfig, StockDataManager, utils


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data management operations for MSc thesis project"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch all stocks command
    fetch_parser = subparsers.add_parser('fetch-all', help='Fetch all configured stocks')
    fetch_parser.add_argument('--period', default='6mo', help='Time period to fetch (default: 6mo)')
    
    # Prepare backtesting command
    backtest_parser = subparsers.add_parser('prepare-backtesting', help='Prepare backtesting dataset')
    
    # Export for CSC command
    export_parser = subparsers.add_parser('export-csc', help='Create CSC deployment package')
    
    # Cache info command
    cache_parser = subparsers.add_parser('cache-info', help='Show cache information')
    
    # Clean cache command
    clean_parser = subparsers.add_parser('clean-cache', help='Clean stale cache files')
    
    # Demonstrate command
    demo_parser = subparsers.add_parser('demo', help='Demonstrate data management system')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize configuration and data manager
    config = ExperimentConfig()
    data_manager = StockDataManager(config)
    
    if args.command == 'fetch-all':
        print(f"Fetching all stocks with period: {args.period}")
        config.data_period = args.period
        results = utils.bulk_fetch_config_stocks(data_manager, config)
        print(f"\nCompleted: {len(results)} stocks successfully fetched")
        
    elif args.command == 'prepare-backtesting':
        print("Preparing backtesting dataset...")
        successful, failed = utils.prepare_backtesting_dataset(data_manager, config)
        print(f"\nCompleted: {successful} successful, {len(failed)} failed")
        
    elif args.command == 'export-csc':
        print("Creating CSC deployment package...")
        backup_file, instructions_file = utils.export_for_csc(data_manager, config)
        if backup_file:
            print(f"\nCSC package ready:")
            print(f"  Backup: {backup_file}")
            print(f"  Instructions: {instructions_file}")
        else:
            print("Failed to create CSC package")
            
    elif args.command == 'cache-info':
        cache_info = data_manager.get_cache_info()
        print(f"\nCache Information:")
        print(f"  Directory: {cache_info['cache_directory']}")
        print(f"  Total Datasets: {cache_info['total_cached_datasets']}")
        print(f"  Fresh Datasets: {cache_info['fresh_datasets']}")
        print(f"  Stale Datasets: {cache_info['stale_datasets']}")
        print(f"  Total Size: {cache_info['total_size_mb']} MB")
        print(f"  Storage Format: {cache_info['storage_format']}")
        
    elif args.command == 'clean-cache':
        print("Cleaning stale cache files...")
        data_manager.clear_stale_cache()
        print("Cache cleaning completed")
        
    elif args.command == 'demo':
        utils.demonstrate_data_management(data_manager)


if __name__ == '__main__':
    main()
