from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Quantum Flow imports
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
from groq import Groq
import networkx as nx
from scipy import signal as scipy_signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Global variables for Quantum Flow Engine
quantum_engine = None
portfolio_manager = None
auto_trader = None
active_connections: List[WebSocket] = []

# Pydantic Models
class QuantumFlowSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    flow_type: str  # WHALE_ACCUMULATION, LIQUIDITY_VACUUM, MOMENTUM_SPIRAL, etc.
    confidence: float
    entry_price: float
    target_multiplier: float
    risk_factor: float
    flow_strength: float
    network_effect: float
    ai_conviction: str
    exit_strategy: Dict
    exchange: str
    status: str = "ACTIVE"
    actual_exit_price: Optional[float] = None
    actual_return: Optional[float] = None
    exit_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MockPortfolioPosition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str
    symbol: str
    flow_type: str
    entry_price: float
    quantity: float  # Mock quantity based on risk amount
    entry_time: datetime
    status: str = "ACTIVE"  # ACTIVE, CLOSED, WATCHING
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MockPortfolio(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "demo_user"  # For demo purposes
    initial_balance: float = 10000.0  # Starting with $10k
    current_balance: float = 10000.0
    positions: List[MockPortfolioPosition] = []
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    active_positions: int = 0
    closed_positions: int = 0
    win_rate: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TradingConfig(BaseModel):
    symbols: List[str]
    max_positions: int = 5
    risk_per_trade: float = 0.02
    min_confidence: float = 0.8
    trading_mode: str = "sandbox"  # or "live"
    update_interval: int = 30

class AutoTraderConfig(BaseModel):
    enabled: bool = False
    min_confidence: float = 0.9  # Higher threshold for auto-trading
    max_positions: int = 3  # Lower limit for auto-trading
    risk_per_trade: float = 0.015  # Lower risk for auto-trading (1.5%)
    max_portfolio_risk: float = 0.05  # Max 5% total portfolio at risk
    profit_target_multiplier: float = 1.1  # Target 10% profit minimum
    stop_loss_multiplier: float = 0.95  # 5% stop loss
    cooldown_minutes: int = 30  # Wait 30 minutes between auto-trades

class PerformanceStats(BaseModel):
    total_signals: int
    profitable_signals: int
    win_rate: float
    avg_return: float
    max_return: float
    min_return: float
    total_return: float
    sharpe_ratio: float

class MarketData(BaseModel):
    symbol: str
    current_price: float
    volume_24h: float
    price_change_24h: float
    market_cap: Optional[float] = None
    timestamp: datetime

class MockPortfolioManager:
    """Manages mock trading portfolio for signal simulation"""
    
    def __init__(self):
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_position_size = 0.1  # Max 10% of portfolio per position
    
    async def get_or_create_portfolio(self, user_id: str = "demo_user") -> MockPortfolio:
        """Get existing portfolio or create new one"""
        try:
            portfolio_data = await db.mock_portfolios.find_one({"user_id": user_id})
            if portfolio_data:
                return MockPortfolio(**portfolio_data)
            else:
                # Create new portfolio
                new_portfolio = MockPortfolio(user_id=user_id)
                await db.mock_portfolios.insert_one(new_portfolio.dict())
                return new_portfolio
        except Exception as e:
            logger.error(f"Error getting/creating portfolio: {e}")
            return MockPortfolio(user_id=user_id)
    
    async def follow_signal(self, signal: QuantumFlowSignal, user_id: str = "demo_user") -> Dict:
        """Follow a signal by creating a mock position"""
        try:
            portfolio = await self.get_or_create_portfolio(user_id)
            
            # Calculate position size based on risk management
            risk_amount = portfolio.current_balance * self.risk_per_trade
            max_position_amount = portfolio.current_balance * self.max_position_size
            
            # Use the smaller of risk-based or max position size
            position_amount = min(risk_amount / signal.risk_factor, max_position_amount)
            quantity = position_amount / signal.entry_price
            
            # Create new position
            position = MockPortfolioPosition(
                signal_id=signal.id,
                symbol=signal.symbol,
                flow_type=signal.flow_type,
                entry_price=signal.entry_price,
                quantity=quantity,
                entry_time=datetime.utcnow(),
                status="ACTIVE"
            )
            
            # Update portfolio
            portfolio.positions.append(position)
            portfolio.active_positions += 1
            portfolio.current_balance -= position_amount  # Reserve the amount
            portfolio.updated_at = datetime.utcnow()
            
            # Save to database
            await db.mock_portfolios.update_one(
                {"user_id": user_id},
                {"$set": portfolio.dict()},
                upsert=True
            )
            
            logger.info(f"📈 Following signal: {signal.symbol} {signal.flow_type} - Amount: ${position_amount:.2f}")
            
            return {
                "success": True,
                "message": f"Now following {signal.symbol} {signal.flow_type} signal",
                "position": position.dict(),
                "portfolio_balance": portfolio.current_balance
            }
            
        except Exception as e:
            logger.error(f"Error following signal: {e}")
            return {
                "success": False,
                "message": f"Failed to follow signal: {str(e)}"
            }
    
    async def watch_signal(self, signal: QuantumFlowSignal, user_id: str = "demo_user") -> Dict:
        """Add signal to watchlist without position"""
        try:
            portfolio = await self.get_or_create_portfolio(user_id)
            
            # Create watch position (no money involved)
            position = MockPortfolioPosition(
                signal_id=signal.id,
                symbol=signal.symbol,
                flow_type=signal.flow_type,
                entry_price=signal.entry_price,
                quantity=0,  # No actual position
                entry_time=datetime.utcnow(),
                status="WATCHING"
            )
            
            portfolio.positions.append(position)
            portfolio.updated_at = datetime.utcnow()
            
            # Save to database
            await db.mock_portfolios.update_one(
                {"user_id": user_id},
                {"$set": portfolio.dict()},
                upsert=True
            )
            
            logger.info(f"👁️ Watching signal: {signal.symbol} {signal.flow_type}")
            
            return {
                "success": True,
                "message": f"Now watching {signal.symbol} {signal.flow_type} signal",
                "position": position.dict()
            }
            
        except Exception as e:
            logger.error(f"Error watching signal: {e}")
            return {
                "success": False,
                "message": f"Failed to watch signal: {str(e)}"
            }
    
    async def update_positions(self, user_id: str = "demo_user"):
        """Update all active positions with current prices"""
        try:
            portfolio = await self.get_or_create_portfolio(user_id)
            updated = False
            
            for position in portfolio.positions:
                if position.status == "ACTIVE" and position.quantity > 0:
                    # Get current price for the symbol
                    if quantum_engine:
                        current_data = await quantum_engine.data_collector.get_market_data(position.symbol)
                        current_price = current_data.get('current_price', position.entry_price)
                        
                        # Calculate P&L
                        pnl = (current_price - position.entry_price) * position.quantity
                        pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
                        
                        # Update position (without modifying the original)
                        position.pnl = pnl
                        position.pnl_percentage = pnl_percentage
                        updated = True
            
            if updated:
                # Recalculate portfolio totals
                portfolio.total_pnl = sum(p.pnl or 0 for p in portfolio.positions if p.status == "ACTIVE")
                portfolio.total_pnl_percentage = (portfolio.total_pnl / portfolio.initial_balance) * 100
                portfolio.updated_at = datetime.utcnow()
                
                # Save to database
                await db.mock_portfolios.update_one(
                    {"user_id": user_id},
                    {"$set": portfolio.dict()},
                    upsert=True
                )
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")

# Global variables for Quantum Flow Engine
quantum_engine = None
portfolio_manager = None
active_connections: List[WebSocket] = []

class RealTimeDataCollector:
    """Enhanced data collector for crypto markets"""
    
    def __init__(self):
        self.exchanges = {}
        self.init_exchanges()
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Coinbase exchange (try without credentials for public data first)
            coinbase_config = {
                'enableRateLimit': True,
            }
            
            # Add credentials if available
            api_key = os.environ.get('COINBASE_API_KEY')
            api_secret = os.environ.get('COINBASE_SECRET')
            
            if api_key and api_secret:
                coinbase_config.update({
                    'apiKey': api_key,
                    'secret': api_secret,
                })
                logger.info(f"✅ Initialized Coinbase exchange with credentials")
            else:
                logger.info(f"✅ Initialized Coinbase exchange in public mode (no credentials)")
            
            self.exchanges['coinbase'] = ccxt.coinbase(coinbase_config)
            trading_mode = os.environ.get('TRADING_MODE', 'sandbox')
            logger.info(f"📊 Trading mode: {trading_mode}")
            if trading_mode == 'sandbox':
                logger.info("📝 Note: Running in demo mode - signals are for analysis only")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize exchanges: {e}")
    
    async def get_market_data(self, symbol: str, exchange_id: str = 'coinbase') -> Dict:
        """Get comprehensive market data"""
        try:
            if exchange_id not in self.exchanges:
                return {}
            
            exchange = self.exchanges[exchange_id]
            
            # Get data in parallel
            ticker_task = exchange.fetch_ticker(symbol)
            orderbook_task = exchange.fetch_order_book(symbol, limit=50)
            trades_task = exchange.fetch_trades(symbol, limit=50)
            ohlcv_task = exchange.fetch_ohlcv(symbol, '1m', limit=60)
            
            ticker, orderbook, trades, ohlcv = await asyncio.gather(
                ticker_task, orderbook_task, trades_task, ohlcv_task,
                return_exceptions=True
            )
            
            # Clean up exception objects before processing
            ticker = ticker if not isinstance(ticker, Exception) else {}
            orderbook = orderbook if not isinstance(orderbook, Exception) else {'bids': [], 'asks': []}
            trades = trades if not isinstance(trades, Exception) else []
            ohlcv = ohlcv if not isinstance(ohlcv, Exception) else []
            
            # Calculate derived metrics with clean data
            liquidity_metrics = self._calculate_liquidity_metrics(orderbook)
            volume_profile = self._calculate_volume_profile(trades)
            price_metrics = self._calculate_price_metrics(ohlcv)
            
            return {
                'symbol': symbol,
                'current_price': ticker.get('last', 0) if ticker else 0,
                'ticker': ticker,
                'orderbook': orderbook,
                'trades': trades,
                'ohlcv': ohlcv,
                'liquidity_metrics': liquidity_metrics,
                'volume_profile': volume_profile,
                'price_metrics': price_metrics,
                'timestamp': datetime.utcnow(),
                'exchange': exchange_id
            }
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            # Return a well-structured empty response instead of empty dict
            return {
                'symbol': symbol,
                'current_price': 0,
                'ticker': {},
                'orderbook': {'bids': [], 'asks': []},
                'trades': [],
                'ohlcv': [],
                'liquidity_metrics': {},
                'volume_profile': {},
                'price_metrics': {},
                'timestamp': datetime.utcnow(),
                'exchange': exchange_id,
                'error': 'Failed to fetch market data'
            }
    
    def _calculate_liquidity_metrics(self, orderbook: Dict) -> Dict:
        """Calculate liquidity metrics from orderbook"""
        try:
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return {}
            
            bids = orderbook['bids'][:20] if orderbook['bids'] else []
            asks = orderbook['asks'][:20] if orderbook['asks'] else []
            
            if not bids or not asks:
                return {}
            
            # Calculate spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
            
            # Calculate depth
            bid_depth = sum(bid[1] * bid[0] for bid in bids)
            ask_depth = sum(ask[1] * ask[0] for ask in asks)
            total_depth = bid_depth + ask_depth
            
            # Calculate imbalance
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            return {
                'spread': spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'imbalance': imbalance
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {
                'spread': 0,
                'bid_depth': 0,
                'ask_depth': 0,
                'total_depth': 0,
                'imbalance': 0
            }
    
    def _calculate_volume_profile(self, trades: List) -> Dict:
        """Calculate volume profile from trades"""
        try:
            if not trades:
                return {}
            
            df = pd.DataFrame(trades)
            if df.empty:
                return {}
            
            # Calculate volume metrics
            total_volume = df['amount'].sum()
            buy_volume = df[df['side'] == 'buy']['amount'].sum() if 'side' in df.columns else total_volume / 2
            sell_volume = total_volume - buy_volume
            
            # Calculate large trade metrics
            if 'price' in df.columns and 'amount' in df.columns:
                trade_values = df['amount'] * df['price']
                large_trade_threshold = trade_values.quantile(0.9) if len(trade_values) > 0 else 0
                large_trades_volume = trade_values[trade_values > large_trade_threshold].sum()
            else:
                large_trades_volume = 0
            
            return {
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.5,
                'large_trades_volume': large_trades_volume,
                'avg_trade_size': df['amount'].mean() if len(df) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {
                'total_volume': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_ratio': 0.5,
                'large_trades_volume': 0,
                'avg_trade_size': 0
            }
    
    def _calculate_price_metrics(self, ohlcv: List) -> Dict:
        """Calculate price metrics from OHLCV data"""
        try:
            if not ohlcv or len(ohlcv) < 10:
                return {}
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate returns and volatility
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(1440)  # Daily volatility
            
            # Calculate momentum
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
            
            # Price acceleration
            price_velocity = df['close'].diff()
            price_acceleration = price_velocity.diff().iloc[-1] if len(price_velocity) > 1 else 0
            
            return {
                'volatility': volatility,
                'momentum': momentum,
                'price_acceleration': price_acceleration,
                'returns_mean': df['returns'].mean(),
                'volume_trend': df['volume'].iloc[-5:].mean() / df['volume'].mean() if len(df) > 5 else 1
            }
        except Exception as e:
            logger.error(f"Error calculating price metrics: {e}")
            return {
                'volatility': 0,
                'momentum': 0,
                'price_acceleration': 0,
                'returns_mean': 0,
                'volume_trend': 1
            }

class QuantumFlowDetector:
    """Enhanced pattern detection engine with AI integration"""
    
    def __init__(self):
        self.groq_client = None
        self.whale_threshold = float(os.environ.get('WHALE_THRESHOLD', 100000))
        self.groq_call_count = 0
        self.last_groq_call_time = datetime.utcnow()
        
        # Groq API optimization - caching and rate limiting
        self.groq_cache = {}  # Cache recent analyses
        self.groq_rate_limit = {}  # Track calls per symbol-pattern
        self.min_confidence_for_groq = float(os.environ.get('MIN_CONFIDENCE_FOR_GROQ', 0.8))
        self.groq_cache_ttl = 300  # 5 minutes cache TTL
        self.max_groq_calls_per_minute = 10  # Rate limit
        
        # Initialize AI client
        groq_key = os.environ.get('GROQ_API_KEY')
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
            logger.info("✅ Initialized Groq AI client with optimization (min confidence: {:.1%}, cache: {}s)".format(
                self.min_confidence_for_groq, self.groq_cache_ttl))
    
    def _should_call_groq_api(self, signal_data: Dict, flow_type: str, confidence: float) -> bool:
        """Determine if Groq API should be called based on optimization rules"""
        if not self.groq_client:
            return False
            
        # Only call for high-confidence signals
        if confidence < self.min_confidence_for_groq:
            return False
            
        # Check rate limiting
        now = datetime.utcnow()
        minute_key = now.strftime('%Y-%m-%d-%H-%M')
        
        if minute_key not in self.groq_rate_limit:
            self.groq_rate_limit = {minute_key: 0}  # Reset counter for new minute
            
        if self.groq_rate_limit[minute_key] >= self.max_groq_calls_per_minute:
            logger.warning(f"⚠️ Groq API rate limit reached for minute {minute_key}")
            return False
            
        # Check cache
        symbol = signal_data.get('symbol', 'UNKNOWN')
        cache_key = f"{symbol}_{flow_type}_{minute_key}"
        
        if cache_key in self.groq_cache:
            cache_time, cached_analysis = self.groq_cache[cache_key]
            if (now - cache_time).total_seconds() < self.groq_cache_ttl:
                logger.info(f"🔄 Using cached Groq analysis for {symbol} {flow_type}")
                return False  # Use cached version
                
        return True
    
    def _get_cached_analysis(self, signal_data: Dict, flow_type: str) -> Optional[Dict]:
        """Get cached analysis if available"""
        symbol = signal_data.get('symbol', 'UNKNOWN') 
        now = datetime.utcnow()
        minute_key = now.strftime('%Y-%m-%d-%H-%M')
        cache_key = f"{symbol}_{flow_type}_{minute_key}"
        
        if cache_key in self.groq_cache:
            cache_time, cached_analysis = self.groq_cache[cache_key]
            if (now - cache_time).total_seconds() < self.groq_cache_ttl:
                cached_analysis['groq_api_called'] = False
                cached_analysis['cached'] = True
                return cached_analysis
                
        return None

    async def _get_ai_analysis(self, signal_data: Dict, flow_type: str, confidence: float) -> Dict:
        """Get comprehensive AI analysis from Groq for signal reasoning with optimization"""
        
        # Default fallback analysis
        fallback_analysis = {
            'ai_conviction': 'MODERATE',
            'market_sentiment': 'NEUTRAL',
            'technical_reasoning': f'{flow_type} detected based on mathematical analysis',
            'risk_assessment': 'Standard risk parameters applied',
            'groq_api_called': False
        }
        
        if not self.groq_client:
            return fallback_analysis
            
        # Check if we should call Groq API
        if not self._should_call_groq_api(signal_data, flow_type, confidence):
            # Try to get cached version
            cached = self._get_cached_analysis(signal_data, flow_type)
            if cached:
                return cached
            return fallback_analysis
        
        try:
            # Update counters
            self.groq_call_count += 1
            self.last_groq_call_time = datetime.utcnow()
            
            # Update rate limiting
            minute_key = datetime.utcnow().strftime('%Y-%m-%d-%H-%M')
            self.groq_rate_limit[minute_key] = self.groq_rate_limit.get(minute_key, 0) + 1
            
            # Prepare market data context for AI analysis
            symbol = signal_data.get('symbol', 'UNKNOWN')
            current_price = signal_data.get('current_price', 0)
            volume_profile = signal_data.get('volume_profile', {})
            liquidity_metrics = signal_data.get('liquidity_metrics', {})
            price_metrics = signal_data.get('price_metrics', {})
            
            # Create comprehensive prompt for AI analysis
            prompt = f"""
            Analyze this {flow_type} trading signal for {symbol}:
            
            MARKET DATA:
            - Current Price: ${current_price:,.2f}
            - Signal Confidence: {confidence:.1%}
            - Buy/Sell Ratio: {volume_profile.get('buy_ratio', 0.5):.2%}
            - Volume Trend: {price_metrics.get('volume_trend', 1):.2f}x
            - Price Momentum: {price_metrics.get('momentum', 0):.2%}
            - Liquidity Spread: {liquidity_metrics.get('spread', 0):.3%}
            - Order Book Imbalance: {liquidity_metrics.get('imbalance', 0):.2%}
            - Large Trades Volume: {volume_profile.get('large_trades_volume', 0):,.0f}
            
            SIGNAL TYPE: {flow_type}
            
            Please provide a concise analysis in exactly this JSON format:
            {{
                "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
                "technical_reasoning": "Brief technical analysis reasoning (max 100 chars)",
                "risk_assessment": "Risk level and key concerns (max 80 chars)",
                "ai_conviction": "HIGH/MODERATE/LOW"
            }}
            
            Focus on:
            1. Market sentiment based on volume and price action
            2. Technical reasoning for this specific pattern
            3. Risk assessment and concerns
            4. Overall conviction level
            """
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert crypto trading analyst. Provide concise, actionable analysis in valid JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse AI response
            ai_content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            try:
                import json
                # Find JSON content between braces
                start_idx = ai_content.find('{')
                end_idx = ai_content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = ai_content[start_idx:end_idx]
                    ai_analysis = json.loads(json_content)
                else:
                    raise ValueError("No JSON found")
                    
            except (json.JSONDecodeError, ValueError):
                # Fallback if JSON parsing fails
                ai_analysis = {
                    'market_sentiment': 'NEUTRAL',
                    'technical_reasoning': f'AI analysis for {flow_type} pattern',
                    'risk_assessment': 'Standard risk parameters',
                    'ai_conviction': 'MODERATE'
                }
            
            # Add metadata
            ai_analysis['groq_api_called'] = True
            ai_analysis['groq_call_timestamp'] = datetime.utcnow().isoformat()
            ai_analysis['cached'] = False
            
            # Cache the result
            cache_key = f"{symbol}_{flow_type}_{minute_key}"
            self.groq_cache[cache_key] = (datetime.utcnow(), ai_analysis.copy())
            
            logger.info(f"🤖 Groq API call #{self.groq_call_count} for {symbol} {flow_type} (confidence: {confidence:.1%}) - Sentiment: {ai_analysis.get('market_sentiment', 'N/A')}")
            
            return ai_analysis
            
        except Exception as e:
            logger.error(f"❌ Groq API call failed: {e}")
            return {
                'ai_conviction': 'MODERATE',
                'market_sentiment': 'NEUTRAL', 
                'technical_reasoning': f'{flow_type} detected via mathematical analysis',
                'risk_assessment': 'Standard risk - AI analysis unavailable',
                'groq_api_called': False,
                'groq_error': str(e)
            }
    
    def get_groq_api_stats(self) -> Dict:
        """Get Groq API usage statistics"""
        return {
            'total_calls': self.groq_call_count,
            'last_call_time': self.last_groq_call_time.isoformat(),
            'calls_per_minute': self.groq_call_count / max(1, (datetime.utcnow() - self.last_groq_call_time).total_seconds() / 60) if self.groq_call_count > 0 else 0
        }

    async def detect_quantum_patterns(self, market_data: Dict) -> List[QuantumFlowSignal]:
        """Main pattern detection method"""
        signals = []
        
        try:
            # Detect different pattern types
            whale_signal = await self._detect_whale_accumulation(market_data)
            if whale_signal:
                signals.append(whale_signal)
            
            liquidity_signal = await self._detect_liquidity_vacuum(market_data)
            if liquidity_signal:
                signals.append(liquidity_signal)
            
            momentum_signal = await self._detect_momentum_spiral(market_data)
            if momentum_signal:
                signals.append(momentum_signal)
            
            volume_signal = await self._detect_volume_anomaly(market_data)
            if volume_signal:
                signals.append(volume_signal)
                
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
        
        return signals
    
    async def _detect_whale_accumulation(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect whale accumulation patterns with AI analysis"""
        try:
            volume_profile = data.get('volume_profile', {})
            liquidity_metrics = data.get('liquidity_metrics', {})
            
            buy_ratio = volume_profile.get('buy_ratio', 0.5)
            large_trades_volume = volume_profile.get('large_trades_volume', 0)
            total_volume = volume_profile.get('total_volume', 1)
            
            # Check for whale accumulation conditions
            if (buy_ratio > 0.7 and 
                large_trades_volume / total_volume > 0.3 and
                liquidity_metrics.get('imbalance', 0) > 0.2):
                
                confidence = min(0.95, 0.7 + buy_ratio * 0.3)
                
                # Get AI analysis for this signal - only if confidence is high enough
                ai_analysis = await self._get_ai_analysis(data, "WHALE_ACCUMULATION", confidence)
                
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="WHALE_ACCUMULATION",
                    confidence=confidence,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.08 + min(0.15, buy_ratio * 0.2),
                    risk_factor=0.04,
                    flow_strength=large_trades_volume / total_volume,
                    network_effect=0.7,
                    ai_conviction=ai_analysis.get('ai_conviction', 'MODERATE'),
                    exit_strategy={
                        'type': 'WHALE_DISTRIBUTION',
                        'stop_loss': 0.03,
                        'profit_target': 0.12,
                        'time_limit': 4,
                        'ai_reasoning': ai_analysis.get('technical_reasoning', 'Whale accumulation pattern detected'),
                        'market_sentiment': ai_analysis.get('market_sentiment', 'NEUTRAL'),
                        'risk_assessment': ai_analysis.get('risk_assessment', 'Standard risk'),
                        'groq_analysis': ai_analysis
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting whale accumulation: {e}")
        
        return None
    
    async def _detect_liquidity_vacuum(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect liquidity vacuum conditions with AI analysis"""
        try:
            liquidity_metrics = data.get('liquidity_metrics', {})
            
            spread = liquidity_metrics.get('spread', 0)
            total_depth = liquidity_metrics.get('total_depth', 0)
            imbalance = abs(liquidity_metrics.get('imbalance', 0))
            
            # Detect thin liquidity conditions
            if spread > 0.002 and total_depth < 50000 and imbalance > 0.3:
                confidence = 0.82
                
                # Get AI analysis for this signal - only if confidence is high enough
                ai_analysis = await self._get_ai_analysis(data, "LIQUIDITY_VACUUM", confidence)
                
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="LIQUIDITY_VACUUM",
                    confidence=confidence,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.05 + min(0.1, spread * 50),
                    risk_factor=0.06,
                    flow_strength=spread,
                    network_effect=0.6,
                    ai_conviction=ai_analysis.get('ai_conviction', 'MODERATE'),
                    exit_strategy={
                        'type': 'LIQUIDITY_RESTORATION',
                        'stop_loss': 0.04,
                        'profit_target': 0.08,
                        'time_limit': 1,
                        'ai_reasoning': ai_analysis.get('technical_reasoning', 'Liquidity vacuum detected'),
                        'market_sentiment': ai_analysis.get('market_sentiment', 'NEUTRAL'),
                        'risk_assessment': ai_analysis.get('risk_assessment', 'Higher risk due to low liquidity'),
                        'groq_analysis': ai_analysis
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting liquidity vacuum: {e}")
        
        return None
    
    async def _detect_momentum_spiral(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect momentum spiral patterns with AI analysis"""
        try:
            price_metrics = data.get('price_metrics', {})
            volume_profile = data.get('volume_profile', {})
            
            momentum = price_metrics.get('momentum', 0)
            price_acceleration = price_metrics.get('price_acceleration', 0)
            volume_trend = price_metrics.get('volume_trend', 1)
            buy_ratio = volume_profile.get('buy_ratio', 0.5)
            
            # Detect momentum acceleration
            if (momentum > 0.02 and 
                price_acceleration > 0.001 and 
                volume_trend > 1.5 and 
                buy_ratio > 0.6):
                
                spiral_strength = momentum * volume_trend * buy_ratio
                confidence = min(0.9, 0.75 + spiral_strength * 2)
                
                # Get AI analysis for this signal - only if confidence is high enough
                ai_analysis = await self._get_ai_analysis(data, "MOMENTUM_SPIRAL", confidence)
                
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="MOMENTUM_SPIRAL",
                    confidence=confidence,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.06 + min(0.12, spiral_strength * 5),
                    risk_factor=0.07,
                    flow_strength=spiral_strength,
                    network_effect=0.8,
                    ai_conviction=ai_analysis.get('ai_conviction', 'HIGH' if spiral_strength > 0.05 else 'MODERATE'),
                    exit_strategy={
                        'type': 'MOMENTUM_EXHAUSTION',
                        'stop_loss': 0.04,
                        'profit_target': 0.10,
                        'time_limit': 2,
                        'ai_reasoning': ai_analysis.get('technical_reasoning', 'Strong momentum spiral detected'),
                        'market_sentiment': ai_analysis.get('market_sentiment', 'BULLISH'),
                        'risk_assessment': ai_analysis.get('risk_assessment', 'Momentum reversal risk'),
                        'groq_analysis': ai_analysis
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting momentum spiral: {e}")
        
        return None
    
    async def _detect_volume_anomaly(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect volume anomaly patterns with AI analysis"""
        try:
            volume_profile = data.get('volume_profile', {})
            price_metrics = data.get('price_metrics', {})
            
            buy_ratio = volume_profile.get('buy_ratio', 0.5)
            volume_trend = price_metrics.get('volume_trend', 1)
            large_trades_ratio = volume_profile.get('large_trades_volume', 0) / max(volume_profile.get('total_volume', 1), 1)
            
            # Detect volume anomalies
            if buy_ratio > 0.75 and volume_trend > 2.0 and large_trades_ratio > 0.4:
                confidence = 0.78
                
                # Get AI analysis for this signal - only if confidence is high enough  
                ai_analysis = await self._get_ai_analysis(data, "VOLUME_ANOMALY", confidence)
                
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="VOLUME_ANOMALY",
                    confidence=confidence,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.04 + min(0.08, (buy_ratio - 0.75) * 0.4),
                    risk_factor=0.08,
                    flow_strength=large_trades_ratio,
                    network_effect=0.5,
                    ai_conviction=ai_analysis.get('ai_conviction', 'MODERATE'),
                    exit_strategy={
                        'type': 'VOLUME_NORMALIZATION',
                        'stop_loss': 0.05,
                        'profit_target': 0.07,
                        'time_limit': 1,
                        'ai_reasoning': ai_analysis.get('technical_reasoning', 'Unusual volume spike detected'),
                        'market_sentiment': ai_analysis.get('market_sentiment', 'NEUTRAL'),
                        'risk_assessment': ai_analysis.get('risk_assessment', 'Volume anomaly risk'),
                        'groq_analysis': ai_analysis
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting volume anomaly: {e}")
        
        return None

class QuantumFlowEngine:
    """Main Quantum Flow trading engine"""
    
    def __init__(self):
        self.data_collector = RealTimeDataCollector()
        self.detector = QuantumFlowDetector()
        self.active_signals = {}
        self.config = TradingConfig(
            symbols=os.environ.get('DEFAULT_SYMBOLS', 'BTC-USD,ETH-USD,SOL-USD').split(','),
            max_positions=int(os.environ.get('MAX_POSITIONS', 5)),
            risk_per_trade=float(os.environ.get('RISK_PER_TRADE', 0.02)),
            min_confidence=float(os.environ.get('MIN_CONFIDENCE', 0.8)),
            trading_mode=os.environ.get('TRADING_MODE', 'sandbox'),
            update_interval=int(os.environ.get('UPDATE_INTERVAL', 30))
        )
        self.running = False
        logger.info("🚀 Quantum Flow Engine initialized")
    
    async def start_analysis(self):
        """Start the main analysis loop"""
        self.running = True
        logger.info("🔄 Starting Quantum Flow analysis loop...")
        
        while self.running:
            try:
                for symbol in self.config.symbols:
                    await self._analyze_symbol(symbol)
                    await asyncio.sleep(1)  # Small delay between symbols
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_symbol(self, symbol: str):
        """Analyze a single trading symbol"""
        try:
            # Collect market data
            market_data = await self.data_collector.get_market_data(symbol)
            
            if not market_data or not market_data.get('current_price'):
                return
            
            # Detect quantum patterns
            signals = await self.detector.detect_quantum_patterns(market_data)
            
            # Filter and process signals
            for signal in signals:
                if await self._should_process_signal(signal):
                    await self._process_new_signal(signal)
            
            # Monitor existing signals
            await self._monitor_active_signals(symbol, market_data.get('current_price', 0))
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    async def _should_process_signal(self, signal: QuantumFlowSignal) -> bool:
        """Check if signal should be processed"""
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check maximum positions
        if len(self.active_signals) >= self.config.max_positions:
            return False
        
        # Check for duplicate signals
        existing_key = f"{signal.symbol}_{signal.flow_type}"
        if existing_key in [f"{s.symbol}_{s.flow_type}" for s in self.active_signals.values()]:
            return False
        
        return True
    
    async def _process_new_signal(self, signal: QuantumFlowSignal):
        """Process and store new signal"""
        try:
            # Store in database
            signal_dict = signal.dict()
            signal_dict['timestamp'] = signal.timestamp.isoformat()
            signal_dict['created_at'] = signal.created_at.isoformat()
            
            await db.signals.insert_one(signal_dict)
            
            # Add to active signals
            self.active_signals[signal.id] = signal
            
            # Log the signal
            logger.info(
                f"🚀 NEW SIGNAL: {signal.flow_type} | {signal.symbol} | "
                f"Confidence: {signal.confidence:.1%} | Target: {signal.target_multiplier:.2f}x"
            )
            
            # AutoTrader evaluation and execution
            global auto_trader
            if auto_trader:
                evaluation = await auto_trader.evaluate_signal(signal)
                
                if evaluation["auto_trade"]:
                    # Execute auto-trade
                    auto_result = await auto_trader.execute_auto_trade(signal)
                    if auto_result.get("success"):
                        logger.info(f"🤖 AUTO-TRADE SUCCESS: {evaluation['reason']}")
                    else:
                        logger.warning(f"🤖 AUTO-TRADE FAILED: {auto_result.get('message', 'Unknown error')}")
                else:
                    logger.debug(f"🤖 AUTO-TRADE SKIPPED: {evaluation['reason']}")
            
            # Broadcast to WebSocket clients
            await self._broadcast_signal(signal)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _monitor_active_signals(self, symbol: str, current_price: float):
        """Monitor active signals for exit conditions"""
        try:
            signals_to_remove = []
            
            for signal_id, signal in self.active_signals.items():
                if signal.symbol != symbol:
                    continue
                
                # Calculate current performance
                current_return = (current_price - signal.entry_price) / signal.entry_price
                time_held = (datetime.utcnow() - signal.timestamp).total_seconds() / 3600
                
                # Check exit conditions
                exit_reason = None
                profit_target = signal.exit_strategy.get('profit_target', 0.1)
                stop_loss = signal.exit_strategy.get('stop_loss', 0.05)
                time_limit = signal.exit_strategy.get('time_limit', 4)
                
                if current_return >= profit_target:
                    exit_reason = "PROFIT_TARGET"
                elif current_return <= -stop_loss:
                    exit_reason = "STOP_LOSS"
                elif time_held >= time_limit:
                    exit_reason = "TIME_LIMIT"
                
                if exit_reason:
                    await self._close_signal(signal, current_price, exit_reason)
                    signals_to_remove.append(signal_id)
            
            # Remove closed signals
            for signal_id in signals_to_remove:
                del self.active_signals[signal_id]
                
        except Exception as e:
            logger.error(f"Error monitoring signals: {e}")
    
    async def _close_signal(self, signal: QuantumFlowSignal, exit_price: float, exit_reason: str):
        """Close a signal and update database"""
        try:
            actual_return = (exit_price - signal.entry_price) / signal.entry_price
            
            # Update in database
            await db.signals.update_one(
                {'id': signal.id},
                {'$set': {
                    'status': 'CLOSED',
                    'actual_exit_price': exit_price,
                    'actual_return': actual_return,
                    'exit_reason': exit_reason
                }}
            )
            
            logger.info(
                f"📊 SIGNAL CLOSED: {signal.flow_type} | {signal.symbol} | "
                f"Return: {actual_return:.2%} | Reason: {exit_reason}"
            )
            
        except Exception as e:
            logger.error(f"Error closing signal: {e}")
    
    async def _broadcast_signal(self, signal: QuantumFlowSignal):
        """Broadcast signal to WebSocket clients"""
        try:
            message = {
                'type': 'new_signal',
                'data': signal.dict()
            }
            
            # Convert datetime objects to strings for JSON serialization
            message['data']['timestamp'] = signal.timestamp.isoformat()
            message['data']['created_at'] = signal.created_at.isoformat()
            
            # Send to all connected clients
            for connection in active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    # Remove dead connections
                    active_connections.remove(connection)
                    
        except Exception as e:
            logger.error(f"Error broadcasting signal: {e}")
    
    async def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """Get performance statistics"""
        try:
            # Query closed signals from last N days
            start_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {
                    '$match': {
                        'status': 'CLOSED',
                        'timestamp': {'$gte': start_date.isoformat()}
                    }
                },
                {
                    '$group': {
                        '_id': None,
                        'total_signals': {'$sum': 1},
                        'profitable_signals': {
                            '$sum': {'$cond': [{'$gt': ['$actual_return', 0]}, 1, 0]}
                        },
                        'avg_return': {'$avg': '$actual_return'},
                        'max_return': {'$max': '$actual_return'},
                        'min_return': {'$min': '$actual_return'},
                        'total_return': {'$sum': '$actual_return'},
                        'returns': {'$push': '$actual_return'}
                    }
                }
            ]
            
            result = await db.signals.aggregate(pipeline).to_list(1)
            
            if result:
                stats = result[0]
                returns = stats.get('returns', [])
                
                # Calculate Sharpe ratio
                if len(returns) > 1:
                    returns_array = np.array(returns)
                    sharpe_ratio = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
                else:
                    sharpe_ratio = 0
                
                return PerformanceStats(
                    total_signals=stats.get('total_signals', 0),
                    profitable_signals=stats.get('profitable_signals', 0),
                    win_rate=stats.get('profitable_signals', 0) / max(stats.get('total_signals', 1), 1),
                    avg_return=stats.get('avg_return', 0),
                    max_return=stats.get('max_return', 0),
                    min_return=stats.get('min_return', 0),
                    total_return=stats.get('total_return', 0),
                    sharpe_ratio=sharpe_ratio
                )
            else:
                return PerformanceStats(
                    total_signals=0, profitable_signals=0, win_rate=0,
                    avg_return=0, max_return=0, min_return=0,
                    total_return=0, sharpe_ratio=0
                )
                
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            return PerformanceStats(
                total_signals=0, profitable_signals=0, win_rate=0,
                avg_return=0, max_return=0, min_return=0,
                total_return=0, sharpe_ratio=0
            )
    
    def stop_analysis(self):
        """Stop the analysis loop"""
        self.running = False
        logger.info("🛑 Quantum Flow analysis stopped")

class AutoTrader:
    """Automated trading agent for high-confidence signals"""
    
    def __init__(self, portfolio_manager, config: AutoTraderConfig = None):
        self.portfolio_manager = portfolio_manager
        self.config = config or AutoTraderConfig()
        self.last_trade_time = {}  # Track cooldowns per symbol
        self.active_auto_positions = {}  # Track auto-positions
        self.total_auto_risk = 0.0
        
        logger.info(f"🤖 AutoTrader initialized - Enabled: {self.config.enabled}")
        
    async def evaluate_signal(self, signal: QuantumFlowSignal) -> Dict:
        """Evaluate if signal meets auto-trading criteria"""
        
        if not self.config.enabled:
            return {"auto_trade": False, "reason": "AutoTrader disabled"}
            
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            return {
                "auto_trade": False, 
                "reason": f"Confidence {signal.confidence:.1%} < {self.config.min_confidence:.1%}"
            }
        
        # Check profit potential
        profit_potential = signal.target_multiplier
        if profit_potential < self.config.profit_target_multiplier:
            return {
                "auto_trade": False,
                "reason": f"Profit potential {profit_potential:.2f} < {self.config.profit_target_multiplier:.2f}"
            }
        
        # Check cooldown
        symbol = signal.symbol
        if symbol in self.last_trade_time:
            time_since_last = (datetime.utcnow() - self.last_trade_time[symbol]).total_seconds() / 60
            if time_since_last < self.config.cooldown_minutes:
                return {
                    "auto_trade": False,
                    "reason": f"Cooldown: {time_since_last:.1f}min < {self.config.cooldown_minutes}min"
                }
        
        # Check position limits
        if len(self.active_auto_positions) >= self.config.max_positions:
            return {
                "auto_trade": False,
                "reason": f"Max positions reached: {len(self.active_auto_positions)}/{self.config.max_positions}"
            }
        
        # Check portfolio risk
        if self.total_auto_risk >= self.config.max_portfolio_risk:
            return {
                "auto_trade": False,
                "reason": f"Max portfolio risk reached: {self.total_auto_risk:.1%}/{self.config.max_portfolio_risk:.1%}"
            }
        
        # Check AI conviction if available
        if hasattr(signal, 'exit_strategy') and signal.exit_strategy.get('groq_analysis'):
            ai_analysis = signal.exit_strategy['groq_analysis']
            ai_conviction = ai_analysis.get('ai_conviction', 'MODERATE')
            if ai_conviction not in ['HIGH', 'VERY_HIGH']:
                return {
                    "auto_trade": False,
                    "reason": f"AI conviction {ai_conviction} not high enough"
                }
        
        return {
            "auto_trade": True,
            "reason": f"✅ Meets all criteria - Confidence: {signal.confidence:.1%}, Profit: {profit_potential:.2f}x"
        }
    
    async def execute_auto_trade(self, signal: QuantumFlowSignal, user_id: str = "auto_trader") -> Dict:
        """Execute automatic trade for qualified signal"""
        
        try:
            # Create auto-trading position with reduced risk
            original_risk = self.portfolio_manager.risk_per_trade
            self.portfolio_manager.risk_per_trade = self.config.risk_per_trade
            
            # Execute the trade
            result = await self.portfolio_manager.follow_signal(signal, user_id)
            
            # Restore original risk
            self.portfolio_manager.risk_per_trade = original_risk
            
            if result.get("success"):
                # Track the auto-position
                position_id = result["position"]["id"]
                self.active_auto_positions[position_id] = {
                    "signal_id": signal.id,
                    "symbol": signal.symbol,
                    "entry_time": datetime.utcnow(),
                    "entry_price": signal.entry_price,
                    "target_price": signal.entry_price * signal.target_multiplier,
                    "stop_loss": signal.entry_price * self.config.stop_loss_multiplier,
                    "risk_amount": result["position"]["quantity"] * signal.entry_price * signal.risk_factor
                }
                
                # Update tracking
                self.last_trade_time[signal.symbol] = datetime.utcnow()
                self.total_auto_risk += self.active_auto_positions[position_id]["risk_amount"]
                
                logger.info(f"🤖 AUTO-TRADE EXECUTED: {signal.symbol} {signal.flow_type} - "
                          f"Confidence: {signal.confidence:.1%}, Target: {signal.target_multiplier:.2f}x")
                
                return {
                    "success": True,
                    "message": f"🤖 Auto-traded {signal.symbol} {signal.flow_type}",
                    "position_id": position_id,
                    "auto_trade": True
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Auto-trade execution error: {e}")
            return {
                "success": False,
                "message": f"Auto-trade failed: {str(e)}",
                "auto_trade": False
            }
    
    async def monitor_positions(self):
        """Monitor auto-positions for exit conditions"""
        
        if not self.active_auto_positions:
            return
            
        try:
            for position_id, position_info in list(self.active_auto_positions.items()):
                # Here you would check current price vs target/stop-loss
                # For now, we'll implement basic time-based exit logic
                
                time_held = (datetime.utcnow() - position_info["entry_time"]).total_seconds() / 3600  # hours
                
                # Auto-exit after 24 hours (can be made configurable)
                if time_held > 24:
                    await self._exit_auto_position(position_id, "Time-based exit")
                    
        except Exception as e:
            logger.error(f"Error monitoring auto-positions: {e}")
    
    async def _exit_auto_position(self, position_id: str, reason: str):
        """Exit an auto-position"""
        
        if position_id in self.active_auto_positions:
            position_info = self.active_auto_positions[position_id]
            
            # Reduce total risk
            self.total_auto_risk -= position_info["risk_amount"]
            
            # Remove from tracking
            del self.active_auto_positions[position_id]
            
            logger.info(f"🤖 AUTO-EXIT: Position {position_id} - {reason}")
    
    def get_status(self) -> Dict:
        """Get current AutoTrader status"""
        
        return {
            "enabled": self.config.enabled,
            "config": self.config.dict(),
            "active_positions": len(self.active_auto_positions),
            "total_risk": self.total_auto_risk,
            "positions": list(self.active_auto_positions.values()),
            "last_trades": {symbol: time.isoformat() for symbol, time in self.last_trade_time.items()}
        }
    
    def update_config(self, new_config: AutoTraderConfig):
        """Update AutoTrader configuration"""
        
        self.config = new_config
        logger.info(f"🤖 AutoTrader config updated - Enabled: {self.config.enabled}")
        return {"success": True, "message": "AutoTrader configuration updated"}

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global quantum_engine, portfolio_manager, auto_trader
    quantum_engine = QuantumFlowEngine()
    portfolio_manager = MockPortfolioManager()
    auto_trader = AutoTrader(portfolio_manager)
    
    # Start analysis in background
    asyncio.create_task(quantum_engine.start_analysis())
    
    yield
    
    # Shutdown
    if quantum_engine:
        quantum_engine.stop_analysis()
        
    # Close exchange connections
    for exchange in quantum_engine.data_collector.exchanges.values():
        try:
            await exchange.close()
        except:
            pass

# Create FastAPI app with lifespan events
app = FastAPI(lifespan=lifespan, title="Quantum Flow Trading API", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Routes
@api_router.get("/")
async def root():
    return {"message": "🚀 Quantum Flow Trading System Online", "status": "active"}

@api_router.get("/signals", response_model=List[QuantumFlowSignal])
async def get_signals(limit: int = 50):
    """Get recent trading signals"""
    try:
        signals = await db.signals.find().sort('created_at', -1).limit(limit).to_list(limit)
        return [QuantumFlowSignal(**signal) for signal in signals]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/signals/active", response_model=List[QuantumFlowSignal])
async def get_active_signals():
    """Get currently active signals"""
    if quantum_engine:
        return list(quantum_engine.active_signals.values())
    return []

@api_router.get("/performance")
async def get_performance(days: int = 30):
    """Get performance statistics"""
    if quantum_engine:
        stats = await quantum_engine.get_performance_stats(days)
        return stats
    return {}

@api_router.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    if quantum_engine:
        data = await quantum_engine.data_collector.get_market_data(symbol)
        return data
    return {}

@api_router.get("/config")
async def get_config():
    """Get current trading configuration"""
    if quantum_engine:
        return quantum_engine.config
    return {}

@api_router.post("/config")
async def update_config(config: TradingConfig):
    """Update trading configuration"""
    if quantum_engine:
        quantum_engine.config = config
        return {"message": "Configuration updated successfully"}
    raise HTTPException(status_code=500, detail="Engine not available")

@api_router.get("/groq-stats")
async def get_groq_stats():
    """Get Groq API usage statistics"""
    if quantum_engine and quantum_engine.detector:
        stats = quantum_engine.detector.get_groq_api_stats()
        return stats
    return {"total_calls": 0, "last_call_time": None, "calls_per_minute": 0}

@api_router.get("/portfolio")
async def get_portfolio(user_id: str = "demo_user"):
    """Get user's mock portfolio"""
    if portfolio_manager:
        portfolio = await portfolio_manager.get_or_create_portfolio(user_id)
        return portfolio
    raise HTTPException(status_code=500, detail="Portfolio manager not available")

@api_router.post("/portfolio/follow/{signal_id}")
async def follow_signal(signal_id: str, user_id: str = "demo_user"):
    """Follow a signal by creating a mock position"""
    if not portfolio_manager:
        raise HTTPException(status_code=500, detail="Portfolio manager not available")
    
    # Get the signal from database
    signal_data = await db.signals.find_one({"id": signal_id})
    if not signal_data:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal = QuantumFlowSignal(**signal_data)
    result = await portfolio_manager.follow_signal(signal, user_id)
    return result

@api_router.post("/portfolio/watch/{signal_id}")
async def watch_signal(signal_id: str, user_id: str = "demo_user"):
    """Watch a signal without creating a position"""
    if not portfolio_manager:
        raise HTTPException(status_code=500, detail="Portfolio manager not available")
    
    # Get the signal from database
    signal_data = await db.signals.find_one({"id": signal_id})
    if not signal_data:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal = QuantumFlowSignal(**signal_data)
    result = await portfolio_manager.watch_signal(signal, user_id)
    return result

@api_router.post("/portfolio/update")
async def update_portfolio_positions(user_id: str = "demo_user"):
    """Update all portfolio positions with current prices"""
    if not portfolio_manager:
        raise HTTPException(status_code=500, detail="Portfolio manager not available")
    
    await portfolio_manager.update_positions(user_id)
    portfolio = await portfolio_manager.get_or_create_portfolio(user_id)
    return {"message": "Portfolio updated successfully", "portfolio": portfolio}

@api_router.get("/portfolio")
async def get_portfolio(user_id: str = "demo_user"):
    """Get user's mock portfolio"""
    if portfolio_manager:
        portfolio = await portfolio_manager.get_or_create_portfolio(user_id)
        return portfolio
    raise HTTPException(status_code=500, detail="Portfolio manager not available")

@api_router.post("/portfolio/follow/{signal_id}")
async def follow_signal(signal_id: str, user_id: str = "demo_user"):
    """Follow a signal by creating a mock position"""
    if not portfolio_manager:
        raise HTTPException(status_code=500, detail="Portfolio manager not available")
    
    # Get the signal from database
    signal_data = await db.signals.find_one({"id": signal_id})
    if not signal_data:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal = QuantumFlowSignal(**signal_data)
    result = await portfolio_manager.follow_signal(signal, user_id)
    return result

@api_router.post("/portfolio/watch/{signal_id}")
async def watch_signal(signal_id: str, user_id: str = "demo_user"):
    """Watch a signal without creating a position"""
    if not portfolio_manager:
        raise HTTPException(status_code=500, detail="Portfolio manager not available")
    
    # Get the signal from database
    signal_data = await db.signals.find_one({"id": signal_id})
    if not signal_data:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    signal = QuantumFlowSignal(**signal_data)
    result = await portfolio_manager.watch_signal(signal, user_id)
    return result

# AutoTrader API endpoints
@api_router.get("/autotrader/status")
async def get_autotrader_status():
    """Get AutoTrader current status and configuration"""
    global auto_trader
    if auto_trader:
        return auto_trader.get_status()
    raise HTTPException(status_code=500, detail="AutoTrader not available")

@api_router.post("/autotrader/config")
async def update_autotrader_config(config: AutoTraderConfig):
    """Update AutoTrader configuration"""
    global auto_trader
    if auto_trader:
        result = auto_trader.update_config(config)
        return result
    raise HTTPException(status_code=500, detail="AutoTrader not available")

@api_router.post("/autotrader/enable")
async def enable_autotrader():
    """Enable AutoTrader"""
    global auto_trader
    if auto_trader:
        auto_trader.config.enabled = True
        return {"message": "AutoTrader enabled", "enabled": True}
    raise HTTPException(status_code=500, detail="AutoTrader not available")

@api_router.post("/autotrader/disable")
async def disable_autotrader():
    """Disable AutoTrader"""
    global auto_trader
    if auto_trader:
        auto_trader.config.enabled = False
        return {"message": "AutoTrader disabled", "enabled": False}
    raise HTTPException(status_code=500, detail="AutoTrader not available")

@api_router.get("/autotrader/positions")
async def get_autotrader_positions():
    """Get AutoTrader active positions"""
    global auto_trader
    if auto_trader:
        return {
            "active_positions": auto_trader.active_auto_positions,
            "total_risk": auto_trader.total_auto_risk,
            "count": len(auto_trader.active_auto_positions)
        }
    raise HTTPException(status_code=500, detail="AutoTrader not available")

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# Include router in app
app.include_router(api_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)