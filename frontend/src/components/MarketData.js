import { useState, useEffect } from 'react';

const MarketData = ({ marketData, fetchMarketData, config }) => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC-USD');
  const [isLoading, setIsLoading] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);

  const symbols = config?.symbols || ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD'];

  useEffect(() => {
    // Fetch initial data for all symbols
    symbols.forEach(symbol => {
      fetchMarketData(symbol);
    });

    // Set up auto-refresh
    const interval = setInterval(() => {
      symbols.forEach(symbol => {
        fetchMarketData(symbol);
      });
    }, 30000); // Refresh every 30 seconds

    setRefreshInterval(interval);

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [config]);

  const handleRefresh = async (symbol = null) => {
    setIsLoading(true);
    try {
      if (symbol) {
        await fetchMarketData(symbol);
      } else {
        // Refresh all symbols
        await Promise.all(symbols.map(sym => fetchMarketData(sym)));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const selectedData = marketData[selectedSymbol] || {};

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">💹 Market Data</h1>
            <p className="text-gray-400">Real-time cryptocurrency market information</p>
          </div>
          <button
            onClick={() => handleRefresh()}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            <span>{isLoading ? '🔄' : '↻'}</span>
            <span>{isLoading ? 'Refreshing...' : 'Refresh All'}</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Symbol Selector */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h2 className="text-lg font-semibold text-white mb-4">📈 Symbols</h2>
            <div className="space-y-2">
              {symbols.map((symbol) => {
                const data = marketData[symbol] || {};
                const displaySymbol = symbol.replace('-USD', '');
                const isActive = symbol === selectedSymbol;
                const price = data.current_price || 0;
                const ticker = data.ticker || {};
                const percentage = ticker.percentage || 0;

                return (
                  <button
                    key={symbol}
                    onClick={() => setSelectedSymbol(symbol)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{displaySymbol}</div>
                        <div className="text-xs opacity-75">
                          {data.current_price 
                            ? `$${data.current_price.toLocaleString()}`
                            : 'Loading...'
                          }
                        </div>
                      </div>
                      {data.ticker?.percentage && (
                        <div className={`text-xs font-medium ${
                          data.ticker.percentage >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {data.ticker.percentage >= 0 ? '+' : ''}{data.ticker.percentage.toFixed(2)}%
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Main Market Data */}
        <div className="lg:col-span-3 space-y-8">
          {/* Price Overview */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">
                💰 {selectedSymbol.replace('-USD', '')} Price Data
              </h2>
              <button
                onClick={() => handleRefresh(selectedSymbol)}
                disabled={isLoading}
                className="text-blue-400 hover:text-blue-300 text-sm"
              >
                {isLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>

            {selectedData.current_price ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">
                    ${selectedData.current_price.toLocaleString('en-US', {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: selectedData.current_price < 1 ? 6 : 2
                    })}
                  </div>
                  <div className="text-gray-400">Current Price</div>
                </div>

                <div className="text-center">
                  <div className={`text-2xl font-bold mb-1 ${
                    selectedData.ticker?.percentage >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {selectedData.ticker?.percentage 
                      ? `${selectedData.ticker.percentage >= 0 ? '+' : ''}${selectedData.ticker.percentage.toFixed(2)}%`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-gray-400">24h Change</div>
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400 mb-1">
                    {selectedData.ticker?.baseVolume 
                      ? `${(selectedData.ticker.baseVolume / 1000000).toFixed(1)}M`
                      : 'N/A'
                    }
                  </div>
                  <div className="text-gray-400">24h Volume</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p className="text-gray-400">Loading market data...</p>
              </div>
            )}
          </div>

          {/* Order Book */}
          {selectedData.orderbook && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">📊 Order Book - Bids</h3>
                <div className="space-y-2">
                  {selectedData.orderbook.bids?.slice(0, 10).map((bid, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-green-900/20 rounded">
                      <span className="text-green-400 font-mono text-sm">
                        ${bid[0].toFixed(2)}
                      </span>
                      <span className="text-white font-mono text-sm">
                        {bid[1].toFixed(4)}
                      </span>
                    </div>
                  )) || <div className="text-gray-400 text-center py-4">No bid data available</div>}
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">📊 Order Book - Asks</h3>
                <div className="space-y-2">
                  {selectedData.orderbook.asks?.slice(0, 10).map((ask, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-red-900/20 rounded">
                      <span className="text-red-400 font-mono text-sm">
                        ${ask[0].toFixed(2)}
                      </span>
                      <span className="text-white font-mono text-sm">
                        {ask[1].toFixed(4)}
                      </span>
                    </div>
                  )) || <div className="text-gray-400 text-center py-4">No ask data available</div>}
                </div>
              </div>
            </div>
          )}

          {/* Market Metrics */}
          {selectedData.liquidity_metrics && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">📈 Market Metrics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  title="Spread"
                  value={`${(selectedData.liquidity_metrics.spread * 100).toFixed(3)}%`}
                  color="blue"
                />
                <MetricCard
                  title="Bid Depth"
                  value={`$${(selectedData.liquidity_metrics.bid_depth / 1000).toFixed(1)}K`}
                  color="green"
                />
                <MetricCard
                  title="Ask Depth"
                  value={`$${(selectedData.liquidity_metrics.ask_depth / 1000).toFixed(1)}K`}
                  color="red"
                />
                <MetricCard
                  title="Imbalance"
                  value={`${(selectedData.liquidity_metrics.imbalance * 100).toFixed(1)}%`}
                  color={selectedData.liquidity_metrics.imbalance > 0 ? 'green' : 'red'}
                />
              </div>
            </div>
          )}

          {/* Trading Activity */}
          {selectedData.trades && selectedData.trades.length > 0 && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">⚡ Recent Trades</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-600">
                      <th className="text-left py-2 text-gray-400">Time</th>
                      <th className="text-right py-2 text-gray-400">Price</th>
                      <th className="text-right py-2 text-gray-400">Amount</th>
                      <th className="text-center py-2 text-gray-400">Side</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedData.trades.slice(0, 10).map((trade, index) => (
                      <tr key={index} className="border-b border-gray-700">
                        <td className="py-2 text-gray-300">
                          {new Date(trade.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="py-2 text-right font-mono text-white">
                          ${trade.price.toFixed(2)}
                        </td>
                        <td className="py-2 text-right font-mono text-white">
                          {trade.amount.toFixed(4)}
                        </td>
                        <td className="py-2 text-center">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            trade.side === 'buy' 
                              ? 'bg-green-600 text-green-100' 
                              : 'bg-red-600 text-red-100'
                          }`}>
                            {trade.side?.toUpperCase()}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Volume Profile */}
          {selectedData.volume_profile && (
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">📊 Volume Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  title="Buy Ratio"
                  value={`${(selectedData.volume_profile.buy_ratio * 100).toFixed(1)}%`}
                  color="green"
                />
                <MetricCard
                  title="Large Trades"
                  value={`$${(selectedData.volume_profile.large_trades_volume / 1000).toFixed(1)}K`}
                  color="purple"
                />
                <MetricCard
                  title="Avg Trade Size"
                  value={`${selectedData.volume_profile.avg_trade_size.toFixed(2)}`}
                  color="blue"
                />
                <MetricCard
                  title="Total Volume"
                  value={`${(selectedData.volume_profile.total_volume / 1000).toFixed(1)}K`}
                  color="yellow"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, color }) => {
  const colorClasses = {
    blue: 'text-blue-400',
    green: 'text-green-400',
    red: 'text-red-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400'
  };

  return (
    <div className="bg-gray-700 rounded-lg p-4 text-center">
      <div className={`text-lg font-bold ${colorClasses[color] || 'text-white'} mb-1`}>
        {value}
      </div>
      <div className="text-xs text-gray-400">{title}</div>
    </div>
  );
};

export default MarketData;