const MarketOverview = ({ marketData }) => {
  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">💹 Market Overview</h3>
      
      <div className="space-y-3">
        {symbols.map((symbol) => {
          const data = marketData[symbol];
          const displaySymbol = symbol.replace('-USD', '');
          
          if (!data || !data.current_price) {
            return (
              <div key={symbol} className="flex items-center justify-between p-2 bg-gray-700 rounded">
                <div>
                  <div className="font-medium text-white">{displaySymbol}</div>
                  <div className="text-xs text-gray-400">Loading...</div>
                </div>
                <div className="animate-pulse bg-gray-600 h-4 w-16 rounded"></div>
              </div>
            );
          }

          const ticker = data.ticker || {};
          const priceChange = ticker.percentage || 0;
          const price = data.current_price;

          return (
            <div key={symbol} className="flex items-center justify-between p-2 bg-gray-700 rounded">
              <div>
                <div className="font-medium text-white">{displaySymbol}</div>
                <div className="text-xs text-gray-400">
                  Vol: {ticker.baseVolume ? `${(ticker.baseVolume / 1000000).toFixed(1)}M` : 'N/A'}
                </div>
              </div>
              <div className="text-right">
                <div className="font-medium text-white">
                  ${price.toLocaleString('en-US', { 
                    minimumFractionDigits: 2, 
                    maximumFractionDigits: price < 1 ? 6 : 2 
                  })}
                </div>
                <div className={`text-xs ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t border-gray-600 text-xs text-gray-400 text-center">
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default MarketOverview;