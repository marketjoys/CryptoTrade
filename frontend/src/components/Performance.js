import { useMemo } from 'react';

const Performance = ({ performance, signals }) => {
  const chartData = useMemo(() => {
    if (!signals || signals.length === 0) return [];

    // Group signals by date and calculate daily performance
    const dailyPerformance = {};
    
    signals
      .filter(signal => signal.status === 'CLOSED' && signal.actual_return)
      .forEach(signal => {
        const date = new Date(signal.timestamp).toISOString().split('T')[0];
        if (!dailyPerformance[date]) {
          dailyPerformance[date] = {
            date,
            totalReturn: 0,
            signalCount: 0,
            profitable: 0
          };
        }
        dailyPerformance[date].totalReturn += signal.actual_return;
        dailyPerformance[date].signalCount += 1;
        if (signal.actual_return > 0) {
          dailyPerformance[date].profitable += 1;
        }
      });

    return Object.values(dailyPerformance).sort((a, b) => 
      new Date(a.date) - new Date(b.date)
    );
  }, [signals]);

  const signalsByType = useMemo(() => {
    const types = {};
    signals.forEach(signal => {
      if (!types[signal.flow_type]) {
        types[signal.flow_type] = {
          count: 0,
          profitable: 0,
          totalReturn: 0,
          avgConfidence: 0
        };
      }
      types[signal.flow_type].count += 1;
      types[signal.flow_type].avgConfidence += signal.confidence;
      
      if (signal.status === 'CLOSED' && signal.actual_return) {
        types[signal.flow_type].totalReturn += signal.actual_return;
        if (signal.actual_return > 0) {
          types[signal.flow_type].profitable += 1;
        }
      }
    });

    // Calculate averages
    Object.keys(types).forEach(type => {
      types[type].avgConfidence /= types[type].count;
      types[type].winRate = types[type].profitable / types[type].count;
    });

    return types;
  }, [signals]);

  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  const formatNumber = (value) => value.toFixed(0);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">📈 Performance Analytics</h1>
        <p className="text-gray-400">Detailed performance metrics and signal analysis</p>
      </div>

      {/* Main Performance Stats */}
      {performance && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Win Rate"
            value={formatPercent(performance.win_rate)}
            subtitle={`${performance.profitable_signals}/${performance.total_signals} profitable`}
            color="green"
          />
          <StatCard
            title="Total Return"
            value={formatPercent(performance.total_return)}
            subtitle="Last 30 days"
            color="blue"
          />
          <StatCard
            title="Average Return"
            value={formatPercent(performance.avg_return)}
            subtitle="Per signal"
            color="purple"
          />
          <StatCard
            title="Sharpe Ratio"
            value={performance.sharpe_ratio.toFixed(2)}
            subtitle="Risk-adjusted return"
            color="yellow"
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Daily Performance Chart */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-6">📊 Daily Performance</h2>
          
          {chartData.length > 0 ? (
            <div className="space-y-3">
              {chartData.slice(-10).map((day, index) => (
                <div key={day.date} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                  <div>
                    <div className="font-medium text-white">
                      {new Date(day.date).toLocaleDateString()}
                    </div>
                    <div className="text-sm text-gray-400">
                      {day.signalCount} signals • {day.profitable} profitable
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-semibold ${
                      day.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPercent(day.totalReturn)}
                    </div>
                    <div className="text-xs text-gray-400">
                      {formatPercent(day.profitable / day.signalCount)} win rate
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <div className="text-4xl mb-4">📊</div>
              <p>No performance data available yet</p>
            </div>
          )}
        </div>

        {/* Signal Type Analysis */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-6">🔬 Signal Type Analysis</h2>
          
          <div className="space-y-4">
            {Object.entries(signalsByType).map(([type, stats]) => (
              <div key={type} className="p-4 bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-medium text-white">
                    {type.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-gray-400">
                    {stats.count} signals
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">Win Rate</div>
                    <div className={`font-medium ${
                      stats.winRate >= 0.6 ? 'text-green-400' : 
                      stats.winRate >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {formatPercent(stats.winRate)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Avg Return</div>
                    <div className={`font-medium ${
                      stats.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPercent(stats.totalReturn / stats.count)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Avg Confidence</div>
                    <div className="font-medium text-white">
                      {formatPercent(stats.avgConfidence)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Best and Worst Performers */}
      {performance && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">🏆 Best Performers</h2>
            <div className="space-y-3">
              {signals
                .filter(s => s.status === 'CLOSED' && s.actual_return)
                .sort((a, b) => b.actual_return - a.actual_return)
                .slice(0, 5)
                .map((signal, index) => (
                  <div key={signal.id} className="flex items-center justify-between p-3 bg-green-900/20 rounded border border-green-700">
                    <div>
                      <div className="font-medium text-white">
                        #{index + 1} {signal.symbol}
                      </div>
                      <div className="text-sm text-gray-400">
                        {signal.flow_type.replace('_', ' ')}
                      </div>
                    </div>
                    <div className="text-green-400 font-semibold">
                      +{formatPercent(signal.actual_return)}
                    </div>
                  </div>
                ))}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h2 className="text-xl font-semibold text-white mb-4">💔 Worst Performers</h2>
            <div className="space-y-3">
              {signals
                .filter(s => s.status === 'CLOSED' && s.actual_return)
                .sort((a, b) => a.actual_return - b.actual_return)
                .slice(0, 5)
                .map((signal, index) => (
                  <div key={signal.id} className="flex items-center justify-between p-3 bg-red-900/20 rounded border border-red-700">
                    <div>
                      <div className="font-medium text-white">
                        #{index + 1} {signal.symbol}
                      </div>
                      <div className="text-sm text-gray-400">
                        {signal.flow_type.replace('_', ' ')}
                      </div>
                    </div>
                    <div className="text-red-400 font-semibold">
                      {formatPercent(signal.actual_return)}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const StatCard = ({ title, value, subtitle, color }) => {
  const colorClasses = {
    green: 'from-green-500 to-green-600',
    blue: 'from-blue-500 to-blue-600',
    purple: 'from-purple-500 to-purple-600',
    yellow: 'from-yellow-500 to-yellow-600'
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <div className="text-center">
        <div className={`text-3xl font-bold bg-gradient-to-r ${colorClasses[color]} bg-clip-text text-transparent mb-2`}>
          {value}
        </div>
        <div className="text-lg font-semibold text-white mb-1">{title}</div>
        <div className="text-sm text-gray-400">{subtitle}</div>
      </div>
    </div>
  );
};

export default Performance;