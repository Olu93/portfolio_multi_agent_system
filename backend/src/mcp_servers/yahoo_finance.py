#!/usr/bin/env python3
"""
yfinance MCP Server
Combines FastMCP with comprehensive financial analysis capabilities
"""

import json
import logging
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import os
import time
from enum import Enum
from typing import Optional, Any, Dict, List
from threading import Thread, Lock
import pandas as pd
import numpy as np
from yfinance import Ticker
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yfinance-mcp-server")
# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8007"))


# Utility Functions
def fetch_ticker(symbol: str) -> Ticker:
    """Helper to safely fetch a yfinance Ticker"""
    return Ticker(symbol.upper())

def safe_get_price(ticker: Ticker) -> float:
    """Safely retrieve current stock price"""
    try:
        # Try current price from info first
        info = ticker.info
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        if price is not None:
            return float(price)
        
        # Fallback to recent history
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        
        raise ValueError("No price data available")
    except Exception as e:
        logger.error(f"Error retrieving price: {e}")
        raise


def validate_ticker(symbol: str) -> bool:
    """Validate if ticker symbol exists"""
    try:
        ticker = fetch_ticker(symbol)
        info = ticker.info
        valid = bool(info and info.get('regularMarketPrice') is not None)
        if not valid:
            logger.debug(f"Ticker '{symbol}' invalid or no price data")
        return valid
    except Exception as e:
        logger.debug(f"Error validating ticker '{symbol}': {e}")
        return False

def format_response(data: Any, success: bool = True, message: str = "") -> Dict[str, Any]:
    """Standardized response format"""
    return {
        'success': success,
        'data': data,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }


class TechnicalIndicators:
    """
    Class that provides various technical indicators and analysis tools for stock data.
    """
    
    @staticmethod
    def get_stock_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve historical stock data for technical analysis.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            ticker = Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            return data
        except Exception as e:
            raise ValueError(f"Error retrieving data for {symbol}: {e}")
    
    @staticmethod
    def calculate_moving_average(data: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate simple moving average.
        
        Args:
            data: DataFrame with price data
            window: Period for moving average
            column: Column name to calculate MA for (default: Close)
            
        Returns:
            Series with moving average values
        """
        return data[column].rolling(window=window).mean()
    
    @staticmethod
    def calculate_exponential_moving_average(data: pd.DataFrame, window: int, column: str = 'Close') -> pd.Series:
        """
        Calculate exponential moving average.
        
        Args:
            data: DataFrame with price data
            window: Period for EMA
            column: Column name to calculate EMA for (default: Close)
            
        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            window: RSI period (default: 14)
            column: Column name to calculate RSI for (default: Close)
            
        Returns:
            Series with RSI values
        """
        delta = data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        
        for i in range(window, len(delta)):
            if i > window:  
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window-1) + gain.iloc[i]) / window
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window-1) + loss.iloc[i]) / window

        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                      signal_period: int = 9, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            column: Column name to calculate MACD for (default: Close)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        fast_ema = TechnicalIndicators.calculate_exponential_moving_average(data, fast_period, column)
        slow_ema = TechnicalIndicators.calculate_exponential_moving_average(data, slow_period, column)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, 
                                num_std: float = 2.0, column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            window: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
            column: Column name for calculation (default: Close)
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands as Series
        """
        middle_band = TechnicalIndicators.calculate_moving_average(data, window, column)
        std_dev = data[column].rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with price data
            window: ATR period (default: 14)
            
        Returns:
            Series with ATR values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20, column: str = 'Close', 
                           annualize: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: DataFrame with price data
            window: Period for volatility calculation (default: 20)
            column: Column name to calculate volatility for (default: Close)
            annualize: Whether to annualize the volatility (default: True)
            
        Returns:
            Series with volatility values
        """
        # Calculate logarithmic returns
        log_returns = np.log(data[column] / data[column].shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=window).std()
        
        # Annualize if requested (assuming 252 trading days)
        if annualize:
            if 'd' in data.index.freq or data.index.freq is None:  # Daily data
                volatility = volatility * np.sqrt(252)
            elif 'h' in data.index.freq:  # Hourly data
                volatility = volatility * np.sqrt(252 * 6.5)  # ~6.5 trading hours per day
            elif 'm' in data.index.freq:  # Minute data
                volatility = volatility * np.sqrt(252 * 6.5 * 60)
                
        return volatility
    
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20, sensitivity: float = 0.03) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using local minima and maxima.
        """
        try:
            if data is None or data.empty or len(data) < (2 * window + 1):
                return {'support': [], 'resistance': []}

            high = data['High']
            low = data['Low']

            resistance_levels = []
            support_levels = []

            for i in range(window, len(high) - window):
                if all(high[i] > high[i - j] for j in range(1, window + 1)) and \
                all(high[i] > high[i + j] for j in range(1, window + 1)):
                    if not any(abs(high[i] - level) / level < sensitivity for level in resistance_levels):
                        resistance_levels.append(high[i])

            for i in range(window, len(low) - window):
                if all(low[i] < low[i - j] for j in range(1, window + 1)) and \
                all(low[i] < low[i + j] for j in range(1, window + 1)):
                    if not any(abs(low[i] - level) / level < sensitivity for level in support_levels):
                        support_levels.append(low[i])

            return {
                'support': sorted(support_levels),
                'resistance': sorted(resistance_levels)
            }

        except Exception as e:
            print(f"[ERROR] detect_support_resistance: {e}")
            return {'support': [], 'resistance': []}

    
    @staticmethod
    def detect_trends(data: pd.DataFrame, short_window: int = 20, long_window: int = 50, 
                    column: str = 'Close') -> Dict[str, pd.Series]:
        """
        Detect trends using moving average crossovers.
        
        Args:
            data: DataFrame with price data
            short_window: Short-term MA period (default: 20)
            long_window: Long-term MA period (default: 50)
            column: Column name to detect trends for (default: Close)
            
        Returns:
            Dictionary with 'trend' and 'signal' Series
        """
        short_ma = TechnicalIndicators.calculate_moving_average(data, short_window, column)
        long_ma = TechnicalIndicators.calculate_moving_average(data, long_window, column)
        
        # Create trend indicator (1: uptrend, -1: downtrend, 0: neutral/undefined)
        trend = pd.Series(0, index=data.index)
        trend[short_ma > long_ma] = 1  # Uptrend
        trend[short_ma < long_ma] = -1  # Downtrend
        
        # Create signal for trend changes
        signal = pd.Series(0, index=data.index)
        signal[(trend.shift(1) <= 0) & (trend > 0)] = 1  # Buy signal (trend turning positive)
        signal[(trend.shift(1) >= 0) & (trend < 0)] = -1  # Sell signal (trend turning negative)
        
        return {
            'trend': trend,
            'signal': signal
        }
    
    @staticmethod
    def calculate_pattern_recognition(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Basic pattern recognition for common candlestick patterns.
        
        Args:
            data: DataFrame with price data (must have Open, High, Low, Close)
            
        Returns:
            Dictionary with pattern signals (1 where pattern is detected)
        """
        pattern_signals = {}
        
        # Doji pattern (open and close are very close)
        doji = pd.Series(0, index=data.index)
        body_size = abs(data['Close'] - data['Open'])
        avg_body = body_size.rolling(window=14).mean()
        shadow_size = data['High'] - data['Low']
        doji[(body_size < 0.1 * shadow_size) & (body_size < 0.25 * avg_body)] = 1
        pattern_signals['doji'] = doji
        
        # Hammer pattern (long lower shadow, small body at the top)
        hammer = pd.Series(0, index=data.index)
        lower_shadow = pd.Series(0, index=data.index)
        upper_shadow = pd.Series(0, index=data.index)
        
        # For days with close > open (bullish)
        bullish = data['Close'] > data['Open']
        lower_shadow[bullish] = data['Open'][bullish] - data['Low'][bullish]
        upper_shadow[bullish] = data['High'][bullish] - data['Close'][bullish]
        
        # For days with open > close (bearish)
        bearish = data['Open'] > data['Close']
        lower_shadow[bearish] = data['Close'][bearish] - data['Low'][bearish]
        upper_shadow[bearish] = data['High'][bearish] - data['Open'][bearish]
        
        # Hammer criteria
        body_height = abs(data['Close'] - data['Open'])
        hammer[(lower_shadow > 2 * body_height) & (upper_shadow < 0.2 * body_height)] = 1
        pattern_signals['hammer'] = hammer
        
        # Engulfing pattern (current candle completely engulfs previous candle)
        bullish_engulfing = pd.Series(0, index=data.index)
        bearish_engulfing = pd.Series(0, index=data.index)
        
        # Bullish engulfing
        bullish_engulfing[(data['Open'] < data['Close'].shift(1)) & 
                         (data['Close'] > data['Open'].shift(1)) &
                         (data['Close'] > data['Open']) &
                         (data['Open'].shift(1) > data['Close'].shift(1))] = 1
        
        # Bearish engulfing
        bearish_engulfing[(data['Open'] > data['Close'].shift(1)) & 
                         (data['Close'] < data['Open'].shift(1)) &
                         (data['Close'] < data['Open']) &
                         (data['Open'].shift(1) < data['Close'].shift(1))] = 1
        
        pattern_signals['bullish_engulfing'] = bullish_engulfing
        pattern_signals['bearish_engulfing'] = bearish_engulfing
        
        return pattern_signals
    
    @staticmethod
    def detect_divergence(data: pd.DataFrame, indicator: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """
        Detect divergence between price and indicator (e.g., RSI).
        
        Args:
            data: DataFrame with price data
            indicator: Series with indicator values (e.g., RSI)
            window: Lookback period for finding pivots (default: 14)
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' Series
        """
        close = data['Close']
        
        bullish_divergence = pd.Series(0, index=data.index)
        bearish_divergence = pd.Series(0, index=data.index)
        
        # Find local price lows and indicator lows
        for i in range(window, len(close) - window):
            # Check for price making lower low
            if (close[i] < close[i-1]) and (close[i] < close[i+1]) and \
               (close[i] < min(close[i-window:i])) and (close[i] < min(close[i+1:i+window+1])):
                
                # But indicator making higher low (bullish divergence)
                if (indicator[i] > indicator[i-window]) and (indicator[i] > indicator[i-window//2]):
                    bullish_divergence[i] = 1
        
        # Find local price highs and indicator highs
        for i in range(window, len(close) - window):
            # Check for price making higher high
            if (close[i] > close[i-1]) and (close[i] > close[i+1]) and \
               (close[i] > max(close[i-window:i])) and (close[i] > max(close[i+1:i+window+1])):
                
                # But indicator making lower high (bearish divergence)
                if (indicator[i] < indicator[i-window]) and (indicator[i] < indicator[i-window//2]):
                    bearish_divergence[i] = 1
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"

# Holder types
class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


# Recommendation types
class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"

class ServerState:
    """Global state manager for MCP server."""
    def __init__(self):
        self.watchlist = set()
        self.watchlist_prices = {}
        self.price_cache = {}
        self.cache_timeout = 300  # seconds
        self.update_thread: Optional[Thread] = None
        self.running = True
        self._lock = Lock()
    
    def add_to_cache(self, symbol: str, data: Any):
        """Add data to cache with timestamp"""
        with self._lock:
            self.price_cache[symbol] = {
                'data': data,
                'timestamp': time.time()
            }
    
    def get_from_cache(self, symbol: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        with self._lock:
            if symbol in self.price_cache:
                cache_entry = self.price_cache[symbol]
                if time.time() - cache_entry['timestamp'] < self.cache_timeout:
                    return cache_entry['data']
                else:
                    del self.price_cache[symbol]
        return None
    
    def cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, value in self.price_cache.items()
                if current_time - value['timestamp'] > self.cache_timeout
            ]
            for key in expired_keys:
                del self.price_cache[key]

# Initialize FastMCP server
mcp = FastMCP(
    "yfinance-production",
    instructions="""
# Yahoo Finance MCP Server

This server provides comprehensive financial market data, stock analysis, and trading insights using Yahoo Finance data.

Available tools:
1. **get_stock_price**: Get current stock price, open/close, day high/low, volume, and market cap.
2. **get_stock_info**: Get detailed stock/company information including sector, industry, P/E ratio, ROE, market cap, etc.
3. **get_historical_stock_prices**: Get historical OHLCV data for a ticker symbol with configurable period and interval.
4. **get_financial_statement**: Get financial statements (annual/quarterly) such as income statement, balance sheet, or cash flow.
5. **get_moving_averages**: Calculate simple and exponential moving averages for multiple window sizes (e.g., 20, 50, 200).
6. **get_rsi**: Calculate the Relative Strength Index (RSI) to identify overbought or oversold market conditions.
7. **get_macd**: Calculate the MACD indicator including MACD line, signal line, and histogram.
8. **get_bollinger_bands**: Calculate Bollinger Bands and identify if the price is within or outside the bands.
9. **get_technical_summary**: Generate a technical summary report combining multiple indicators (RSI, MACD, MAs, Bollinger, volatility, support/resistance).
10. **add_to_watchlist**: Add a stock ticker to the user’s local watchlist for price tracking.
11. **remove_from_watchlist**: Remove a stock ticker from the watchlist.
12. **get_watchlist**: Retrieve the full list of tickers currently in the watchlist.
13. **get_watchlist_prices**: Get current prices of all stocks in the watchlist.
14. **get_realtime_watchlist_prices**: Get cached real-time price updates for all watchlist stocks.
15. **get_yahoo_finance_news**: Get the latest Yahoo Finance news articles related to a specific ticker.
16. **get_recommendations**: Fetch recent analyst ratings, upgrades, downgrades, and recommendations for a stock.
17. **compare_stocks**: Compare two stocks by price, difference, percentage difference, and identify which is higher.

Each tool returns structured JSON data and is optimized for real-time financial insights.
""",
    host=MCP_HOST, port=MCP_PORT
)



state = ServerState()
ti = TechnicalIndicators()


@mcp.tool("get_stock_price")
def get_stock_price(symbol: str) -> Dict[str, Any]:

    """
    Retrieve the current stock price and related data for the given symbol, 
    utilizing an internal cache to minimize redundant API calls.

    Args:
        symbol (str): The ticker symbol of the stock (case-insensitive).

    Returns:
        Dict[str, Any]: A JSON object containing:
            - symbol (str): The stock symbol in uppercase.
            - current_price (float): The latest stock price, rounded to 2 decimals.
            - previous_close (float): Previous closing price.
            - open (float): Opening price for the current trading day.
            - day_high (float): Highest price of the day.
            - day_low (float): Lowest price of the day.
            - volume (int): Trading volume.
            - market_cap (int): Market capitalization.
            - currency (str): Trading currency (default 'USD').
            
    """
    symbol = symbol.upper()
    
    # Attempt to retrieve cached price data to reduce API calls
    cached_price = state.get_from_cache(f"price_{symbol}")
    if cached_price:
        return format_response(cached_price)
    
    try:
        ticker = fetch_ticker(symbol)
        price = safe_get_price(ticker)
        info = ticker.info
        
        price_data = {
            'symbol': symbol,
            'current_price': round(price, 2),
            'previous_close': round(info.get('previousClose', 0), 2),
            'open': round(info.get('regularMarketOpen', 0), 2),
            'day_high': round(info.get('dayHigh', 0), 2),
            'day_low': round(info.get('dayLow', 0), 2),
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
        
       
        state.add_to_cache(f"price_{symbol}", price_data)
        
        return format_response(price_data)
    except Exception as e:
        return format_response(None, False, str(e))



@mcp.tool("get_stock_info")
def get_stock_info(ticker: str) -> Dict[str, Any]:

    """
    Retrieve comprehensive company information for a given stock ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (case-insensitive).

    Returns:
        str: A JSON-formatted string containing company details such as:
            - symbol (str): Stock symbol in uppercase.
            - long_name (str): Full company name.
            - short_name (str): Shortened company name.
            - sector (str): Sector classification.
            - industry (str): Industry classification.
            - country (str): Country of origin.
            - website (str): Company website URL.
            - business_summary (str): Truncated business summary (up to 500 characters).
            - employees (int): Number of full-time employees.
            - market_cap (int): Market capitalization.
            - enterprise_value (int): Enterprise value.
            - pe_ratio (float): Trailing Price to Earnings ratio.
            - forward_pe (float): Forward Price to Earnings ratio.
            - price_to_book (float): Price to Book ratio.
            - debt_to_equity (float): Debt to Equity ratio.
            - return_on_equity (float): Return on Equity ratio.
            - currency (str): Trading currency (default 'USD').

    """
    ticker = ticker.upper()
    
    try:
        company = Ticker(ticker)
        info = company.info
        
        if not info or not info.get('longName'):
            return format_response(None, False, f"Ticker {ticker} not found")
        
        company_data = {
            'symbol': ticker,
            'long_name': info.get('longName', 'N/A'),
            'short_name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'business_summary': info.get('businessSummary', 'N/A')[:500] + "..." if info.get('businessSummary') else 'N/A',
            'employees': info.get('fullTimeEmployees', 0),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'price_to_book': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'return_on_equity': info.get('returnOnEquity', 0),
            'currency': info.get('currency', 'USD')
        }
        
        return format_response(company_data)
    except Exception as e:
        return format_response(None, False, str(e))



@mcp.tool("get_historical_stock_prices")
def get_historical_stock_prices(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:

    """
    Retrieve historical stock price data for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (case-insensitive).
        period (str, optional): The time period to retrieve data for (e.g., '1mo', '3mo', '1y'). Defaults to "1mo".
        interval (str, optional): The data interval (e.g., '1d', '1wk', '1mo'). Defaults to "1d".

    Returns:
        Dict[str, Any]: JSON object containing:
            - symbol (str): The ticker symbol.
            - period (str): The requested historical period.
            - interval (str): The data interval.
            - data (list): List of daily price data dicts with fields:
                - date (str): Date in YYYY-MM-DD format.
                - open (float): Opening price.
                - high (float): Highest price.
                - low (float): Lowest price.
                - close (float): Closing price.
                - volume (int): Trading volume.
            - total_records (int): Number of records returned.

    """
    ticker = ticker.upper()
    
    try:
        company = Ticker(ticker)
        data = company.history(period=period, interval=interval)
        
        if data.empty:
            return format_response(None, False, f"No historical data found for {ticker}")
        
        # Convert to list of dictionaries
        historical_data = []
        for date, row in data.iterrows():
            historical_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume'])
            })
        
        result = {
            'symbol': ticker,
            'period': period,
            'interval': interval,
            'data': historical_data,
            'total_records': len(historical_data)
        }
        
        return format_response(result)
    except Exception as e:
        return format_response(None, False, str(e))



@mcp.tool("get_financial_statement")
def get_financial_statement(ticker: str, financial_type: str) -> Dict[str, Any]:

    """
    Retrieve specified financial statements for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (case-insensitive).
        financial_type (str): Type of financial statement to retrieve. Expected values correspond to 
                              FinancialType enum keys, such as:
                              - income_stmt
                              - quarterly_income_stmt
                              - balance_sheet
                              - quarterly_balance_sheet
                              - cashflow
                              - quarterly_cashflow

    Returns:
        Dict[str, Any]: JSON object containing:
            - symbol (str): The ticker symbol.
            - financial_type (str): The requested financial statement type.
            - data (list): A list of dictionaries, each representing the financial data for a specific date.
                           Each dictionary contains:
                               - date (str): The statement date (YYYY-MM-DD).
                               - financial metrics (key-value pairs): Various financial metrics with numeric values or None.

    """
    ticker = ticker.upper()
    
    try:
        company = Ticker(ticker)
        
        mapping = {
            FinancialType.income_stmt: company.income_stmt,
            FinancialType.quarterly_income_stmt: company.quarterly_income_stmt,
            FinancialType.balance_sheet: company.balance_sheet,
            FinancialType.quarterly_balance_sheet: company.quarterly_balance_sheet,
            FinancialType.cashflow: company.cashflow,
            FinancialType.quarterly_cashflow: company.quarterly_cashflow,
        }
        
        fs_data = mapping.get(financial_type)
        if fs_data is None or fs_data.empty:
            return format_response(None, False, f"No {financial_type} data available for {ticker}")
        
        # Convert dataframe to list of dicts with date keys
        result = []
        for column in fs_data.columns:
            date_str = column.strftime("%Y-%m-%d") if isinstance(column, pd.Timestamp) else str(column)
            row = {"date": date_str}
            for idx, value in fs_data[column].items():
                row[idx] = None if pd.isna(value) else float(value) if isinstance(value, (int, float)) else value
            result.append(row)
        
        return format_response({
            'symbol': ticker,
            'financial_type': financial_type,
            'data': result
        })
    except Exception as e:
        return format_response(None, False, str(e))


@mcp.tool("get_holder_info")
async def get_holder_info(ticker: str, holder_type: str) -> Dict[str, Any]:
    """Get holder information for a given ticker symbol."""
    try:
        company = Ticker(ticker.upper())
        if company.isin is None:
            return format_response(None, False, f"Company ticker '{ticker}' not found.")
    except Exception as e:
        return format_response(None, False, f"Error getting holder info for '{ticker}': {e}")

    try:
        match holder_type:
            case HolderType.major_holders:
                data = company.major_holders.reset_index(names="metric").to_dict(orient="records")
            case HolderType.institutional_holders:
                data = company.institutional_holders.to_dict(orient="records")
            case HolderType.mutualfund_holders:
                data = company.mutualfund_holders.to_dict(orient="records")
            case HolderType.insider_transactions:
                data = company.insider_transactions.to_dict(orient="records")
            case HolderType.insider_purchases:
                data = company.insider_purchases.to_dict(orient="records")
            case HolderType.insider_roster_holders:
                data = company.insider_roster_holders.to_dict(orient="records")
            case _:
                return format_response(
                    None,
                    False,
                    f"Error: Invalid holder type '{holder_type}'. Valid types: {', '.join([h.value for h in HolderType])}"
                )

        return format_response(data)
    except Exception as e:
        return format_response(None, False, f"Error retrieving {holder_type} data for '{ticker}': {e}")



# Technical Analysis Tools
@mcp.tool("get_moving_averages")
def get_moving_averages(symbol: str, period: str = "6mo", interval: str = "1d", 
                        windows: List[int] = [20, 50, 200]) -> Dict[str, Any]:
    
    """Calculate multiple Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) for a stock."""
    try:
        data = ti.get_stock_data(symbol, period, interval)
        result = {}
        
        for window in windows:
            sma = ti.calculate_moving_average(data, window)
            ema = ti.calculate_exponential_moving_average(data, window)
            
            result[f'SMA_{window}'] = sma.dropna().round(2).tolist()
            result[f'EMA_{window}'] = ema.dropna().round(2).tolist()
        
        result.update({
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'close': data['Close'].round(2).tolist(),
            'symbol': symbol.upper()
        })
        
        return format_response(result)
    except Exception as e:
        return format_response(None, False, str(e))


@mcp.tool("get_rsi")
def get_rsi(symbol: str, period: str = "6mo", interval: str = "1d", window: int = 14) -> Dict[str, Any]:
    """ Calculate the Relative Strength Index (RSI) for a given stock symbol."""
    try:
        data = ti.get_stock_data(symbol, period, interval)
        rsi = ti.calculate_rsi(data, window)
        
        result = {
            'symbol': symbol.upper(),
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'rsi': rsi.dropna().round(2).tolist(),
            'close': data['Close'].round(2).tolist(),
            'current_rsi': round(rsi.iloc[-1], 2),
            'overbought': bool(rsi.iloc[-1] > 70), 
            'oversold': bool(rsi.iloc[-1] < 30)     
        }
        
        return format_response(result)
    except Exception as e:
        return format_response(None, False, str(e))


@mcp.tool("get_macd")
def get_macd(symbol: str, period: str = "6mo", interval: str = "1d", 
            fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
    
    """Calculate the Moving Average Convergence Divergence (MACD) indicator for a given stock symbol."""
    try:
        data = ti.get_stock_data(symbol, period, interval)
        macd_data = ti.calculate_macd(data, fast_period, slow_period, signal_period)
        
        result = {
            'symbol': symbol.upper(),
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'macd': macd_data['macd'].dropna().round(4).tolist(),
            'signal': macd_data['signal'].dropna().round(4).tolist(),
            'histogram': macd_data['histogram'].dropna().round(4).tolist(),
            'close': data['Close'].round(2).tolist(),
            'bullish_crossover': bool(macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1])
        }
        
        return format_response(result)
    except Exception as e:
        return format_response(None, False, str(e))



@mcp.tool("get_bollinger_bands")
def get_bollinger_bands(symbol: str, period: str = "6mo", interval: str = "1d",
                        window: int = 20, num_std: float = 2.0) -> Dict[str, Any]:
    
    """ Calculate Bollinger Bands for a given stock symbol.

    Bollinger Bands consist of a middle band (simple moving average) and upper and lower bands
    calculated based on standard deviations from the middle band."""
    try:
        data = ti.get_stock_data(symbol, period, interval)
        bb_data = ti.calculate_bollinger_bands(data, window, num_std)
        
        current_price = data['Close'].iloc[-1]
        upper_band = bb_data['upper'].iloc[-1]
        lower_band = bb_data['lower'].iloc[-1]
        
        result = {
            'symbol': symbol.upper(),
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'upper': bb_data['upper'].dropna().round(2).tolist(),
            'middle': bb_data['middle'].dropna().round(2).tolist(),
            'lower': bb_data['lower'].dropna().round(2).tolist(),
            'close': data['Close'].round(2).tolist(),
            'price_position': 'above_upper' if current_price > upper_band else 'below_lower' if current_price < lower_band else 'within_bands',
            'squeeze': bool((upper_band - lower_band) / bb_data['middle'].iloc[-1] < 0.1)
        }
        
        return format_response(result)
    except Exception as e:
        return format_response(None, False, str(e))
    

    
@mcp.tool("get_technical_summary")
def get_technical_summary(symbol: str) -> Dict[str, Any]:
    """
    Generate a comprehensive technical analysis summary for a given stock symbol.

    This function calculates popular technical indicators and provides a summary
    including trend signals, support/resistance levels, and volatility.

    Indicators:
    - SMA (20, 50, 200)
    - RSI
    - MACD
    - Bollinger Bands
    - Annualized volatility
    - Support/resistance detection
    """
    try:
        data = ti.get_stock_data(symbol, period="1y", interval="1d")

        if data is None or data.empty:
            return format_response(None, False, f"No data found for {symbol.upper()}")

        if len(data) < 200:
            return format_response(None, False, "Not enough data to calculate indicators (need 200+ days)")

        latest_price = data['Close'].iloc[-1]

        # Calculate indicators
        sma_20 = ti.calculate_moving_average(data, 20).iloc[-1]
        sma_50 = ti.calculate_moving_average(data, 50).iloc[-1]
        sma_200 = ti.calculate_moving_average(data, 200).iloc[-1]

        rsi = ti.calculate_rsi(data).iloc[-1]

        macd_data = ti.calculate_macd(data)
        macd = macd_data['macd'].iloc[-1]
        macd_signal = macd_data['signal'].iloc[-1]

        bb_data = ti.calculate_bollinger_bands(data)
        bb_upper = bb_data['upper'].iloc[-1]
        bb_lower = bb_data['lower'].iloc[-1]
        bb_middle = bb_data['middle'].iloc[-1]

        volatility = ti.calculate_volatility(data).iloc[-1]

        # Detect support/resistance levels
        levels = ti.detect_support_resistance(data)
        if not levels or not isinstance(levels, dict) or \
           'support' not in levels or 'resistance' not in levels:
            print(f"[WARN] Missing or invalid support/resistance data for {symbol.upper()}")
            levels = {'support': [], 'resistance': []}

        # Generate analysis signals
        signals = []

        # Moving average trend
        if latest_price > sma_20 > sma_50 > sma_200:
            signals.append("Strong bullish trend - all MAs aligned")
        elif latest_price > sma_20:
            signals.append("Short-term bullish - price above SMA(20)")
        elif latest_price < sma_20:
            signals.append("Short-term bearish - price below SMA(20)")

        # RSI
        if rsi > 70:
            signals.append("Overbought condition - RSI > 70")
        elif rsi < 30:
            signals.append("Oversold condition - RSI < 30")

        # MACD
        if macd > macd_signal:
            signals.append("MACD bullish - above signal line")
        else:
            signals.append("MACD bearish - below signal line")

        # Bollinger Bands
        if latest_price > bb_upper:
            signals.append("Price above upper Bollinger Band")
        elif latest_price < bb_lower:
            signals.append("Price below lower Bollinger Band")

        # Trend assessment
        bullish_signals = sum(1 for s in signals if isinstance(s, str) and "bullish" in s.lower())
        bearish_signals = sum(1 for s in signals if isinstance(s, str) and "bearish" in s.lower())

        if bullish_signals > bearish_signals:
            overall_trend = "Bullish"
        elif bearish_signals > bullish_signals:
            overall_trend = "Bearish"
        else:
            overall_trend = "Neutral"

        # Final result
        result = {
            "symbol": symbol.upper(),
            "last_price": round(latest_price, 2),
            "overall_trend": overall_trend,
            "signals": signals,
            "indicators": {
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "sma_200": round(sma_200, 2),
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "macd_signal": round(macd_signal, 4),
                "bb_upper": round(bb_upper, 2),
                "bb_middle": round(bb_middle, 2),
                "bb_lower": round(bb_lower, 2),
                "volatility_annualized": round(volatility * 100, 2)
            },
            "support_resistance": {
                "resistance_levels": levels["resistance"][:3],
                "support_levels": levels["support"][:3]
            }
        }

        return format_response(result)

    except Exception as e:
        return format_response(None, False, str(e))



# Watchlist Management
@mcp.tool("add_to_watchlist")
def add_to_watchlist(symbol: str) -> Dict[str, Any]:

    """Add a stock symbol to the user's watchlist."""
    symbol = symbol.upper()
    
    if not validate_ticker(symbol):
        return format_response(None, False, f"Invalid ticker symbol: {symbol}")
    
    state.watchlist.add(symbol)
    return format_response(
        {'symbol': symbol, 'watchlist_size': len(state.watchlist)},
        True,
        f"Added {symbol} to watchlist"
    )

@mcp.tool("remove_from_watchlist")
def remove_from_watchlist(symbol: str) -> Dict[str, Any]:

    """Remove a stock symbol from the user's watchlist."""
    symbol = symbol.upper()
    
    if symbol in state.watchlist:
        state.watchlist.remove(symbol)
        return format_response(
            {'symbol': symbol, 'watchlist_size': len(state.watchlist)},
            True,
            f"Removed {symbol} from watchlist"
        )
    
    return format_response(None, False, f"{symbol} not in watchlist")


@mcp.tool("get_watchlist")
def get_watchlist() -> Dict[str, Any]:

    """Retrieve all stock symbols currently in the watchlist."""
    return format_response({
        'symbols': sorted(list(state.watchlist)),
        'count': len(state.watchlist)
    })


@mcp.tool("get_watchlist_prices")
def get_watchlist_prices() -> Dict[str, Any]:

    """Fetch the current prices for all stocks in the watchlist"""
    if not state.watchlist:
        return format_response([], True, "Watchlist is empty")
    
    prices = []
    for symbol in sorted(state.watchlist):
        try:
            ticker = fetch_ticker(symbol)
            price = safe_get_price(ticker)
            prices.append({
                'symbol': symbol,
                'price': round(price, 2),
                'status': 'success'
            })
        except Exception as e:
            prices.append({
                'symbol': symbol,
                'price': None,
                'status': 'error',
                'error': str(e)
            })
    
    return format_response(prices)


# News and Recommendations
@mcp.tool("get_yahoo_finance_news")
def get_yahoo_finance_news(ticker: str) -> Dict[str, Any]:

    """Get the latest news articles related to a stock ticker from Yahoo Finance."""
    ticker = ticker.upper()
    
    try:
        company = Ticker(ticker)
        news_list = company.news
        
        results = []
        for i in news_list[:10]:  # Limit to 10 most recent
            item: dict = i.get('content')
            results.append({
                'title': item.get('title', ''),
                'publisher': item.get('provider', {}).get('url', ''),
                'link': item.get('canonicalUrl', {}).get('url', ''),
                'published': item.get('pubDate', ''),
                'summary': item.get('summary', '')
            })
        
        return format_response({
            'symbol': ticker,
            'news': results,
            'count': len(results)
        })
    except Exception as e:
        return format_response(None, False, str(e))

@mcp.tool("get_recommendations")
async def get_recommendations(
    ticker: str,
    recommendation_type: str,
    months_back: int = 12
) -> Dict[str, Any]:
    
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""

    company = Ticker(ticker.upper())

    try:
        if company.isin is None:
            return format_response([], False, f"Company ticker '{ticker}' not found.")

    except Exception as e:
        return format_response([], False, f"Error accessing company info for '{ticker}': {e}")

    try:
        if recommendation_type == RecommendationType.recommendations:
            recs = company.recommendations
            if recs is None or recs.empty:
                return format_response([], True, "No recommendations found.")

            data = recs.reset_index().to_dict(orient="records")
            return format_response(data)

        elif recommendation_type == RecommendationType.upgrades_downgrades:
            upgrades = company.upgrades_downgrades
            if upgrades is None or upgrades.empty:
                return format_response([], True, "No upgrades/downgrades found.")

            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades = upgrades.reset_index()
            upgrades_filtered = upgrades[upgrades["GradeDate"] >= cutoff_date]
            upgrades_sorted = upgrades_filtered.sort_values("GradeDate", ascending=False)
            latest_by_firm = upgrades_sorted.drop_duplicates(subset=["Firm"])

            data = latest_by_firm.to_dict(orient="records")
            return format_response(data)

        else:
            return format_response([], False, f"Invalid recommendation_type '{recommendation_type}'.")

    except Exception as e:
        return format_response([], False, f"Error getting recommendations for '{ticker}': {e}")



@mcp.tool("compare_stocks")
def compare_stocks(symbol1: str, symbol2: str) -> Dict[str, Any]:

    """Compare two stocks"""
    symbol1, symbol2 = symbol1.upper(), symbol2.upper()
    
    try:
        price1 = safe_get_price(fetch_ticker(symbol1))
        price2 = safe_get_price(fetch_ticker(symbol2))
        
        comparison = {
            'symbol1': {'symbol': symbol1, 'price': round(price1, 2)},
            'symbol2': {'symbol': symbol2, 'price': round(price2, 2)},
            'difference': round(price1 - price2, 2),
            'percentage_difference': round(((price1 - price2) / price2) * 100, 2),
            'higher': symbol1 if price1 > price2 else symbol2 if price2 > price1 else 'equal'
        }
        
        return format_response(comparison)
    except Exception as e:
        return format_response(None, False, str(e))

# Background price update function
def update_watchlist_prices():

    """Background thread to update watchlist prices"""
    while state.running:
        try:
            for symbol in list(state.watchlist):
                try:
                    ticker = fetch_ticker(symbol)
                    price = safe_get_price(ticker)
                    state.watchlist_prices[symbol] = {
                        'price': round(price, 2),
                        'updated': datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error updating price for {symbol}: {e}")
                    state.watchlist_prices[symbol] = {
                        'price': None,
                        'error': str(e),
                        'updated': datetime.now().isoformat()
                    }
            
            # Cleanup cache
            state.cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error in price update thread: {e}")
        
        time.sleep(60)  # Update every minute




if __name__ == "__main__":
    print("=== Starting YFinance MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== YFinance MCP Server shutting down ===")     

