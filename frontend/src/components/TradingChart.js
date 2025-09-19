import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, HistogramSeries } from 'lightweight-charts';

const TradingChart = ({ symbol, signals = [], onSymbolChange }) => {
  const chartContainerRef = useRef();
  const chart = useRef();
  const candleSeries = useRef();
  const volumeSeries = useRef();
  const [isLoading, setIsLoading] = useState(true);
  const [chartData, setChartData] = useState(null);
  const [currentSymbol, setCurrentSymbol] = useState(symbol || 'BTC-USD');

  const SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'];

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    chart.current = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1f2937' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
        secondsVisible: false,
      },
      watermark: {
        color: '#374151',
        visible: true,
        text: 'Quantum Flow Trading',
        fontSize: 24,
        horzAlign: 'left',
        vertAlign: 'bottom',
      },
    });

    // Add candlestick series using v5 API with series classes
    candleSeries.current = chart.current.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    // Add volume series using v5 API with series classes
    volumeSeries.current = chart.current.addSeries(HistogramSeries, {
      color: '#6b7280',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });

    chart.current.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.7,
        bottom: 0,
      },
    });

    // Resize handler
    const handleResize = () => {
      if (chart.current && chartContainerRef.current) {
        chart.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Load initial data
    loadChartData(currentSymbol);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chart.current) {
        chart.current.remove();
      }
    };
  }, []);

  useEffect(() => {
    if (currentSymbol !== symbol && symbol) {
      setCurrentSymbol(symbol);
      loadChartData(symbol);
    }
  }, [symbol]);

  useEffect(() => {
    // Add signal markers when signals change
    if (signals && signals.length > 0 && chart.current) {
      addSignalMarkers(signals);
    }
  }, [signals]);

  const loadChartData = async (symbolToLoad) => {
    setIsLoading(true);
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/market-data/${symbolToLoad}`);
      const data = await response.json();
      
      if (data && data.error) {
        console.error('Market data error:', data.error);
        setIsLoading(false);
        return;
      }

      // Convert OHLCV data to chart format
      const ohlcvData = data.ohlcv || [];
      if (ohlcvData.length > 0) {
        const candleData = ohlcvData.map(item => ({
          time: Math.floor(item[0] / 1000), // Convert to seconds
          open: item[1],
          high: item[2],
          low: item[3],
          close: item[4],
        }));

        const volumeData = ohlcvData.map(item => ({
          time: Math.floor(item[0] / 1000),
          value: item[5],
          color: item[4] >= item[1] ? '#10b98150' : '#ef444450', // Green for up, red for down
        }));

        // Update chart series
        if (candleSeries.current && volumeSeries.current) {
          candleSeries.current.setData(candleData);
          volumeSeries.current.setData(volumeData);
          
          // Fit content
          chart.current.timeScale().fitContent();
        }

        setChartData(data);
      }
    } catch (error) {
      console.error('Error loading chart data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const addSignalMarkers = (signalsToAdd) => {
    if (!candleSeries.current) return;

    const markers = signalsToAdd
      .filter(signal => signal.symbol === currentSymbol)
      .map(signal => {
        const signalColors = {
          'WHALE_ACCUMULATION': '#3b82f6',
          'LIQUIDITY_VACUUM': '#8b5cf6',
          'MOMENTUM_SPIRAL': '#10b981',
          'VOLUME_ANOMALY': '#f59e0b'
        };

        const signalIcons = {
          'WHALE_ACCUMULATION': '🐋',
          'LIQUIDITY_VACUUM': '💫',
          'MOMENTUM_SPIRAL': '🌪️',
          'VOLUME_ANOMALY': '📊'
        };

        return {
          time: Math.floor(new Date(signal.timestamp).getTime() / 1000),
          position: 'belowBar',
          color: signalColors[signal.flow_type] || '#6b7280',
          shape: 'arrowUp',
          text: `${signalIcons[signal.flow_type] || '📈'} ${signal.flow_type} (${(signal.confidence * 100).toFixed(0)}%)`,
        };
      });

    candleSeries.current.setMarkers(markers);
  };

  const handleSymbolChange = (newSymbol) => {
    setCurrentSymbol(newSymbol);
    loadChartData(newSymbol);
    if (onSymbolChange) {
      onSymbolChange(newSymbol);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const formatVolume = (volume) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toFixed(0);
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-bold text-white">Live Trading Chart</h2>
          
          {/* Symbol Selector */}
          <select
            value={currentSymbol}
            onChange={(e) => handleSymbolChange(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
          >
            {SYMBOLS.map(sym => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>
        </div>

        {/* Market Data Info */}
        <div className="flex items-center space-x-4 text-sm">
          {chartData && (
            <>
              <div className="text-white">
                <span className="text-gray-400">Price: </span>
                <span className="font-medium">{formatPrice(chartData.current_price || 0)}</span>
              </div>
              <div className="text-white">
                <span className="text-gray-400">24h Volume: </span>
                <span className="font-medium">{formatVolume(chartData.ticker?.baseVolume || 0)}</span>
              </div>
              <div className={`${
                (chartData.ticker?.percentage || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                <span className="text-gray-400">24h Change: </span>
                <span className="font-medium">{(chartData.ticker?.percentage || 0).toFixed(2)}%</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center z-10">
            <div className="text-white text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div>
              <div>Loading {currentSymbol} chart data...</div>
            </div>
          </div>
        )}
        
        <div
          ref={chartContainerRef}
          className="w-full h-96 bg-gray-900 rounded"
          style={{ minHeight: '400px' }}
        />
      </div>

      {/* Chart Legend */}
      <div className="mt-4 flex items-center justify-between text-xs text-gray-400">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>Bullish Candle</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Bearish Candle</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-gray-500 rounded"></div>
            <span>Volume</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <span>🐋 Whale | 💫 Liquidity | 🌪️ Momentum | 📊 Volume</span>
        </div>
      </div>

      {/* Signal Summary */}
      {signals && signals.length > 0 && (
        <div className="mt-4 p-3 bg-gray-700 rounded">
          <div className="text-sm text-gray-300 mb-2">
            Active Signals for {currentSymbol}: {signals.filter(s => s.symbol === currentSymbol && s.status === 'ACTIVE').length}
          </div>
          <div className="flex flex-wrap gap-2">
            {signals
              .filter(s => s.symbol === currentSymbol && s.status === 'ACTIVE')
              .slice(0, 3)
              .map(signal => (
                <div key={signal.id} className="text-xs bg-gray-600 px-2 py-1 rounded">
                  {signal.flow_type} ({(signal.confidence * 100).toFixed(0)}%)
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingChart;