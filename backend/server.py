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

class TradingConfig(BaseModel):
    symbols: List[str]
    max_positions: int = 5
    risk_per_trade: float = 0.02
    min_confidence: float = 0.8
    trading_mode: str = "sandbox"  # or "live"
    update_interval: int = 30

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

class RealTimeDataCollector:
    """Enhanced data collector for crypto markets"""
    
    def __init__(self):
        self.exchanges = {}
        self.init_exchanges()
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Coinbase exchange (no sandbox mode available)
            coinbase_config = {
                'apiKey': os.environ.get('COINBASE_API_KEY'),
                'secret': os.environ.get('COINBASE_SECRET'),
                'enableRateLimit': True,
            }
            
            if coinbase_config['apiKey']:
                self.exchanges['coinbase'] = ccxt.coinbase(coinbase_config)
                trading_mode = os.environ.get('TRADING_MODE', 'sandbox')
                logger.info(f"✅ Initialized Coinbase exchange (mode: {trading_mode})")
                if trading_mode == 'sandbox':
                    logger.info("📝 Note: Running in demo mode - signals are for analysis only")
            else:
                logger.warning("⚠️ No Coinbase API credentials found - running in demo mode only")
            
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
            
            # Calculate derived metrics
            liquidity_metrics = self._calculate_liquidity_metrics(orderbook)
            volume_profile = self._calculate_volume_profile(trades)
            price_metrics = self._calculate_price_metrics(ohlcv)
            
            return {
                'symbol': symbol,
                'current_price': ticker.get('last', 0) if not isinstance(ticker, Exception) else 0,
                'ticker': ticker if not isinstance(ticker, Exception) else {},
                'orderbook': orderbook if not isinstance(orderbook, Exception) else {},
                'trades': trades if not isinstance(trades, Exception) else [],
                'ohlcv': ohlcv if not isinstance(ohlcv, Exception) else [],
                'liquidity_metrics': liquidity_metrics,
                'volume_profile': volume_profile,
                'price_metrics': price_metrics,
                'timestamp': datetime.utcnow(),
                'exchange': exchange_id
            }
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return {}
    
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
            return {}
    
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
            return {}
    
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
            return {}

class QuantumFlowDetector:
    """Enhanced pattern detection engine"""
    
    def __init__(self):
        self.groq_client = None
        self.whale_threshold = float(os.environ.get('WHALE_THRESHOLD', 100000))
        
        # Initialize AI client
        groq_key = os.environ.get('GROQ_API_KEY')
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
            logger.info("✅ Initialized Groq AI client")
    
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
        """Detect whale accumulation patterns"""
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
                    ai_conviction="HIGH" if confidence > 0.85 else "MODERATE",
                    exit_strategy={
                        'type': 'WHALE_DISTRIBUTION',
                        'stop_loss': 0.03,
                        'profit_target': 0.12,
                        'time_limit': 4
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting whale accumulation: {e}")
        
        return None
    
    async def _detect_liquidity_vacuum(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect liquidity vacuum conditions"""
        try:
            liquidity_metrics = data.get('liquidity_metrics', {})
            
            spread = liquidity_metrics.get('spread', 0)
            total_depth = liquidity_metrics.get('total_depth', 0)
            imbalance = abs(liquidity_metrics.get('imbalance', 0))
            
            # Detect thin liquidity conditions
            if spread > 0.002 and total_depth < 50000 and imbalance > 0.3:
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="LIQUIDITY_VACUUM",
                    confidence=0.82,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.05 + min(0.1, spread * 50),
                    risk_factor=0.06,
                    flow_strength=spread,
                    network_effect=0.6,
                    ai_conviction="MODERATE",
                    exit_strategy={
                        'type': 'LIQUIDITY_RESTORATION',
                        'stop_loss': 0.04,
                        'profit_target': 0.08,
                        'time_limit': 1
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting liquidity vacuum: {e}")
        
        return None
    
    async def _detect_momentum_spiral(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect momentum spiral patterns"""
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
                
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="MOMENTUM_SPIRAL",
                    confidence=min(0.9, 0.75 + spiral_strength * 2),
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.06 + min(0.12, spiral_strength * 5),
                    risk_factor=0.07,
                    flow_strength=spiral_strength,
                    network_effect=0.8,
                    ai_conviction="HIGH" if spiral_strength > 0.05 else "MODERATE",
                    exit_strategy={
                        'type': 'MOMENTUM_EXHAUSTION',
                        'stop_loss': 0.04,
                        'profit_target': 0.10,
                        'time_limit': 2
                    },
                    exchange=data.get('exchange', 'coinbase')
                )
        except Exception as e:
            logger.error(f"Error detecting momentum spiral: {e}")
        
        return None
    
    async def _detect_volume_anomaly(self, data: Dict) -> Optional[QuantumFlowSignal]:
        """Detect volume anomaly patterns"""
        try:
            volume_profile = data.get('volume_profile', {})
            price_metrics = data.get('price_metrics', {})
            
            buy_ratio = volume_profile.get('buy_ratio', 0.5)
            volume_trend = price_metrics.get('volume_trend', 1)
            large_trades_ratio = volume_profile.get('large_trades_volume', 0) / max(volume_profile.get('total_volume', 1), 1)
            
            # Detect volume anomalies
            if buy_ratio > 0.75 and volume_trend > 2.0 and large_trades_ratio > 0.4:
                return QuantumFlowSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    timestamp=datetime.utcnow(),
                    flow_type="VOLUME_ANOMALY",
                    confidence=0.78,
                    entry_price=data.get('current_price', 0),
                    target_multiplier=1.04 + min(0.08, (buy_ratio - 0.75) * 0.4),
                    risk_factor=0.08,
                    flow_strength=large_trades_ratio,
                    network_effect=0.5,
                    ai_conviction="MODERATE",
                    exit_strategy={
                        'type': 'VOLUME_NORMALIZATION',
                        'stop_loss': 0.05,
                        'profit_target': 0.07,
                        'time_limit': 1
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

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global quantum_engine
    quantum_engine = QuantumFlowEngine()
    
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