import { useState, useEffect } from 'react';
import SignalCard from './SignalCard';
import PerformanceCard from './PerformanceCard';
import MarketOverview from './MarketOverview';
import GroqStats from './GroqStats';

const Dashboard = ({ signals, activeSignals, performance, marketData, isConnected }) => {
  const [stats, setStats] = useState({
    totalSignals: 0,
    activeSignals: 0,
    todaySignals: 0,
    winRate: 0
  });

  useEffect(() => {
    calculateStats();
  }, [signals, activeSignals, performance]);

  const calculateStats = () => {
    const today = new Date().toDateString();
    const todaySignals = signals.filter(signal => 
      new Date(signal.timestamp).toDateString() === today
    ).length;

    setStats({
      totalSignals: signals.length,
      activeSignals: activeSignals.length,
      todaySignals,
      winRate: performance ? performance.win_rate : 0
    });
  };

  const recentSignals = signals.slice(0, 5);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">
          🚀 Quantum Flow Dashboard
        </h1>
        <p className="text-gray-400">
          Real-time crypto trading signals powered by AI
        </p>
      </div>

      {/* Status Banner */}
      <div className={`mb-8 p-4 rounded-lg border ${
        isConnected 
          ? 'bg-green-900/20 border-green-600 text-green-100' 
          : 'bg-red-900/20 border-red-600 text-red-100'
      }`}>
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full mr-3 ${
            isConnected ? 'bg-green-400' : 'bg-red-400'
          }`}></div>
          <span className="font-medium">
            {isConnected 
              ? '🔴 System is LIVE - Monitoring markets in real-time' 
              : '⚠️ System disconnected - Attempting to reconnect...'}
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Signals"
          value={stats.totalSignals}
          icon="📊"
          color="blue"
        />
        <StatCard
          title="Active Signals"
          value={stats.activeSignals}
          icon="🚀"
          color="green"
        />
        <StatCard
          title="Today's Signals"
          value={stats.todaySignals}
          icon="⚡"
          color="purple"
        />
        <StatCard
          title="Win Rate"
          value={`${(stats.winRate * 100).toFixed(1)}%`}
          icon="🎯"
          color="yellow"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Signals */}
        <div className="lg:col-span-2">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">
                🔥 Latest Signals
              </h2>
              <div className="text-sm text-gray-400">
                Last {recentSignals.length} signals
              </div>
            </div>
            
            {recentSignals.length > 0 ? (
              <div className="space-y-4">
                {recentSignals.map((signal) => (
                  <SignalCard key={signal.id} signal={signal} />
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <div className="text-4xl mb-4">⏳</div>
                <p>No signals detected yet. System is analyzing markets...</p>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Performance Card */}
          <PerformanceCard performance={performance} />
          
          {/* Groq AI Stats */}
          <GroqStats />
          
          {/* Market Overview */}
          <MarketOverview marketData={marketData} />
          
          {/* Active Signals Summary */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              🎯 Active Positions
            </h3>
            {activeSignals.length > 0 ? (
              <div className="space-y-3">
                {activeSignals.slice(0, 3).map((signal) => (
                  <div key={signal.id} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <div className="font-medium text-white">{signal.symbol}</div>
                      <div className="text-xs text-gray-400">{signal.flow_type}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-green-400">
                        {(signal.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-400">
                        {signal.target_multiplier.toFixed(2)}x
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <div className="text-2xl mb-2">💤</div>
                <p className="text-sm">No active positions</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon, color }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    purple: 'from-purple-500 to-purple-600',
    yellow: 'from-yellow-500 to-yellow-600'
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <div className="flex items-center">
        <div className={`w-12 h-12 bg-gradient-to-r ${colorClasses[color]} rounded-lg flex items-center justify-center text-white text-xl`}>
          {icon}
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;