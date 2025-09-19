import { useState, useEffect } from 'react';

const MarketOverview = ({ marketData, config }) => {
  const symbols = config?.symbols || ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD'];
  const [loadingStates, setLoadingStates] = useState({});
  const [lastUpdateTime, setLastUpdateTime] = useState(new Date());

  useEffect(() => {
    // Initialize loading states for all symbols
    const initialLoadingStates = {};
    symbols.forEach(symbol => {
      initialLoadingStates[symbol] = !marketData[symbol] || !marketData[symbol].current_price;
    });
    setLoadingStates(initialLoadingStates);
    setLastUpdateTime(new Date());
  }, [marketData]);

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">💹 Market Overview</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-gray-400">Live</span>
        </div>
      </div>
      
      <div className="space-y-3">
        {symbols.map((symbol) => {
          const data = marketData[symbol];
          const displaySymbol = symbol.replace('-USD', '');
          const isLoading = loadingStates[symbol] || (!data || !data.current_price);
          
          if (isLoading) {
            return (
              <div key={symbol} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                    <span className="text-white text-sm font-bold">
                      {displaySymbol.substring(0, 1)}
                    </span>
                  </div>
                  <div>
                    <div className="font-medium text-white">{displaySymbol}</div>
                    <div className="text-xs text-gray-400 flex items-center space-x-1">
                      <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                      <span>Loading market data...</span>
                    </div>
                  </div>
                </div>
                <div className="flex flex-col items-end space-y-1">
                  <div className="animate-pulse bg-gray-600 h-5 w-20 rounded"></div>
                  <div className="animate-pulse bg-gray-600 h-3 w-12 rounded"></div>
                </div>
              </div>
            );
          }

          const ticker = data.ticker || {};
          const priceChange = ticker.percentage || 0;
          const price = data.current_price;

          // Get crypto-specific colors
          const cryptoColors = {
            'BTC': 'from-orange-400 to-orange-600',
            'ETH': 'from-blue-400 to-purple-600', 
            'SOL': 'from-purple-400 to-pink-600',
            'XRP': 'from-blue-500 to-cyan-500'
          };

          return (
            <div key={symbol} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg hover:bg-gray-650 transition-colors">
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 bg-gradient-to-r ${cryptoColors[displaySymbol] || 'from-gray-500 to-gray-600'} rounded-full flex items-center justify-center`}>
                  <span className="text-white text-sm font-bold">
                    {displaySymbol.substring(0, displaySymbol === 'XRP' ? 3 : 1)}
                  </span>
                </div>
                <div>
                  <div className="font-medium text-white">{displaySymbol}</div>
                  <div className="text-xs text-gray-400">
                    Vol: {ticker.baseVolume ? `${(ticker.baseVolume / 1000000).toFixed(1)}M` : 'N/A'}
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="font-medium text-white">
                  ${price.toLocaleString('en-US', { 
                    minimumFractionDigits: 2, 
                    maximumFractionDigits: price < 1 ? 6 : 2 
                  })}
                </div>
                <div className={`text-xs flex items-center ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  <span className="mr-1">
                    {priceChange >= 0 ? '↗' : '↘'}
                  </span>
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-gray-600 text-xs text-gray-400 text-center flex items-center justify-center space-x-2">
        <div className="w-1.5 h-1.5 bg-green-400 rounded-full"></div>
        <span>Last updated: {lastUpdateTime.toLocaleTimeString()}</span>
      </div>
    </div>
  );
};

export default MarketOverview;