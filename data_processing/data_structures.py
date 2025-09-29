"""
Core data structures for the MSc thesis project.

This module defines the fundamental data types used throughout the project
for stock data representation and trading signals.
"""

import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalType(Enum):
    """Trading signal types supported by the system."""
    THREE_CLASS = "3class"  # Buy/Hold/Sell
    FIVE_CLASS = "5class"   # Strong Buy/Buy/Hold/Sell/Strong Sell


class ModelProvider(Enum):
    """Model provider types for forecasting."""
    AZURE = "azure"
    CSC = "csc"
    LOCAL = "local"


@dataclass
class TradingSignal:
    """
    Represents a trading signal prediction with metadata.
    
    Attributes:
        symbol: Stock symbol (e.g., "AAPL")
        signal: Predicted signal (e.g., "BUY", "HOLD", "SELL")
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Optional explanation for the signal
        timestamp: When the prediction was made
        model_used: Which model generated the prediction
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
    """
    symbol: str
    signal: str
    confidence: float
    reasoning: Optional[str] = None
    timestamp: datetime = None
    model_used: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class StockData:
    """
    Container for stock market OHLC data.
    
    Attributes:
        symbol: Stock symbol (e.g., "AAPL")
        ohlc: DataFrame with Open, High, Low, Close columns and datetime index
    """
    symbol: str
    ohlc: pd.DataFrame
    
    def get_ohlc_sequence(self, days: int = 30) -> str:
        """
        Format OHLC data as time series string for LLM input.
        
        Args:
            days: Number of recent days to include
            
        Returns:
            Formatted string with OHLC data for each date
        """
        recent_data = self.ohlc.tail(days)
        lines = []
        for date, row in recent_data.iterrows():
            lines.append(
                f"{date.strftime('%Y-%m-%d')}: "
                f"O={row['Open']:.2f} H={row['High']:.2f} "
                f"L={row['Low']:.2f} C={row['Close']:.2f}"
            )
        return "\n".join(lines)
    
    def get_latest_price(self) -> float:
        """Get the most recent closing price."""
        if self.ohlc.empty:
            raise ValueError(f"No data available for {self.symbol}")
        return self.ohlc['Close'].iloc[-1]
    
    def get_price_change(self, days: int = 1) -> float:
        """
        Calculate price change over specified days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Price change as a decimal (e.g., 0.05 for 5% increase)
        """
        if len(self.ohlc) < days + 1:
            raise ValueError(f"Insufficient data for {days}-day price change")
        
        current_price = self.ohlc['Close'].iloc[-1]
        previous_price = self.ohlc['Close'].iloc[-(days + 1)]
        
        return (current_price - previous_price) / previous_price
    
    def get_volatility(self, days: int = 30) -> float:
        """
        Calculate rolling volatility over specified days.
        
        Args:
            days: Number of days for volatility calculation
            
        Returns:
            Annualized volatility
        """
        if len(self.ohlc) < days:
            raise ValueError(f"Insufficient data for {days}-day volatility")
        
        returns = self.ohlc['Close'].pct_change().dropna()
        recent_returns = returns.tail(days)
        
        return recent_returns.std() * (252 ** 0.5)  # Annualized
    
    def summary_stats(self) -> dict:
        """Get summary statistics for the stock data."""
        if self.ohlc.empty:
            return {"error": "No data available"}
        
        return {
            "symbol": self.symbol,
            "data_points": len(self.ohlc),
            "date_range": {
                "start": self.ohlc.index.min().strftime('%Y-%m-%d'),
                "end": self.ohlc.index.max().strftime('%Y-%m-%d')
            },
            "price_stats": {
                "latest_close": self.ohlc['Close'].iloc[-1],
                "min_price": self.ohlc['Low'].min(),
                "max_price": self.ohlc['High'].max(),
                "avg_close": self.ohlc['Close'].mean()
            }
        }


@dataclass
class ModelConfig:
    """
    Configuration for AI models used in forecasting.
    
    Attributes:
        provider: Model provider (Azure, CSC, Local)
        model_name: Name of the specific model
        api_endpoint: API endpoint URL (if applicable)
        api_key: API key for authentication (if applicable)
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens in response
    """
    provider: ModelProvider
    model_name: str
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150
