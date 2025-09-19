const PerformanceCard = ({ performance }) => {
  if (!performance) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">📈 Performance</h3>
        <div className="text-center py-4 text-gray-400">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p className="text-sm">Loading performance data...</p>
        </div>
      </div>
    );
  }

  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  const formatNumber = (value) => value.toFixed(0);

  const getPerformanceColor = (value) => {
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">📈 Performance (30d)</h3>
      
      <div className="space-y-4">
        {/* Win Rate */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Win Rate</span>
          <span className={`font-semibold ${getPerformanceColor(performance.win_rate - 0.5)}`}>
            {formatPercent(performance.win_rate)}
          </span>
        </div>

        {/* Total Return */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Total Return</span>
          <span className={`font-semibold ${getPerformanceColor(performance.total_return)}`}>
            {formatPercent(performance.total_return)}
          </span>
        </div>

        {/* Average Return */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Avg Return</span>
          <span className={`font-semibold ${getPerformanceColor(performance.avg_return)}`}>
            {formatPercent(performance.avg_return)}
          </span>
        </div>

        {/* Signal Count */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Total Signals</span>
          <span className="font-semibold text-white">
            {formatNumber(performance.total_signals)}
          </span>
        </div>

        {/* Profitable Signals */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Profitable</span>
          <span className="font-semibold text-green-400">
            {formatNumber(performance.profitable_signals)}
          </span>
        </div>

        {/* Sharpe Ratio */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Sharpe Ratio</span>
          <span className={`font-semibold ${getPerformanceColor(performance.sharpe_ratio)}`}>
            {performance.sharpe_ratio.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-xs text-gray-400 text-center">
          {performance.total_signals > 0 ? (
            <>
              <div className="mb-1">
                Best: <span className="text-green-400">{formatPercent(performance.max_return)}</span>
              </div>
              <div>
                Worst: <span className="text-red-400">{formatPercent(performance.min_return)}</span>
              </div>
            </>
          ) : (
            <div>No completed signals yet</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PerformanceCard;