import { useState, useEffect } from 'react';
import SignalCard from './SignalCard';
import PerformanceCard from './PerformanceCard';
import MarketOverview from './MarketOverview';
import GroqStats from './GroqStats';
import TradingChart from './TradingChart';
import PortfolioDashboard from './PortfolioDashboard';

const Dashboard = ({ signals, activeSignals, performance, marketData, isConnected, config }) => {
  const [stats, setStats] = useState({
    totalSignals: 0,
    activeSignals: 0,
    todaySignals: 0,
    winRate: 0
  });
  const [currentView, setCurrentView] = useState('overview'); // overview, charts, portfolio
  const [selectedSymbol, setSelectedSymbol] = useState('BTC-USD');

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

  const handleFollowSignal = async (signal) => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/portfolio/follow/${signal.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      
      if (result.success) {
        alert(`✅ ${result.message}\nPortfolio Balance: $${result.portfolio_balance?.toFixed(2) || 'N/A'}`);
      } else {
        alert(`❌ ${result.message}`);
      }
    } catch (error) {
      console.error('Error following signal:', error);
      alert('❌ Failed to follow signal. Please try again.');
    }
  };

  const handleWatchSignal = async (signal) => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/portfolio/watch/${signal.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();
      
      if (result.success) {
        alert(`✅ ${result.message}`);
      } else {
        alert(`❌ ${result.message}`);
      }
    } catch (error) {
      console.error('Error watching signal:', error);
      alert('❌ Failed to watch signal. Please try again.');
    }
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

      {/* Navigation Tabs */}
      <div className="mb-8">
        <div className="flex space-x-4 border-b border-gray-700">
          <button
            onClick={() => setCurrentView('overview')}
            className={`pb-2 px-1 text-sm font-medium border-b-2 transition-colors ${
              currentView === 'overview'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-gray-400 hover:text-gray-300'
            }`}
          >
            📊 Overview
          </button>
          <button
            onClick={() => setCurrentView('charts')}
            className={`pb-2 px-1 text-sm font-medium border-b-2 transition-colors ${
              currentView === 'charts'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-gray-400 hover:text-gray-300'
            }`}
          >
            📈 Live Charts
          </button>
          <button
            onClick={() => setCurrentView('portfolio')}
            className={`pb-2 px-1 text-sm font-medium border-b-2 transition-colors ${
              currentView === 'portfolio'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-gray-400 hover:text-gray-300'
            }`}
          >
            💼 Portfolio
          </button>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`mb-8 p-4 rounded-lg border ${
        isConnected 
          ? 'bg-green-900/20 border-green-600 text-green-100' 
          : 'bg-red-900/20 border-red-600 text-red-100'
      }`}>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
          <span className="font-medium">
            {isConnected ? '🟢 System Online' : '🔴 Connection Lost'}
          </span>
          <span className="text-sm opacity-75">
            | {stats.totalSignals} Total Signals | {stats.activeSignals} Active | {stats.todaySignals} Today
          </span>
        </div>
      </div>

      {/* Content Based on Selected View */}
      {currentView === 'overview' && (
        <>
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">{stats.totalSignals}</div>
              <div className="text-sm text-gray-400">Total Signals</div>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">{stats.activeSignals}</div>
              <div className="text-sm text-gray-400">Active Now</div>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
              <div className="text-3xl font-bold text-yellow-400 mb-2">{stats.todaySignals}</div>
              <div className="text-sm text-gray-400">Today</div>
            </div>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">{(stats.winRate * 100).toFixed(1)}%</div>
              <div className="text-sm text-gray-400">Win Rate</div>
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column */}
            <div className="lg:col-span-2 space-y-8">
              {/* Market Overview */}
              <MarketOverview marketData={marketData} />
              
              {/* Performance */}
              <PerformanceCard performance={performance} />
            </div>

            {/* Right Sidebar */}
            <div className="space-y-8">
              {/* Groq Stats */}
              <GroqStats />
              
              {/* Recent Signals */}
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-white mb-4">🔥 Recent Signals</h3>
                <div className="space-y-4">
                  {recentSignals.length > 0 ? (
                    recentSignals.map((signal) => (
                      <SignalCard 
                        key={signal.id} 
                        signal={signal} 
                        showActions={true}
                        onFollowSignal={handleFollowSignal}
                        onWatchSignal={handleWatchSignal}
                      />
                    ))
                  ) : (
                    <div className="text-center py-4">
                      <div className="text-gray-400 mb-2">📡</div>
                      <div className="text-sm text-gray-400">Scanning for signals...</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {currentView === 'charts' && (
        <div className="space-y-8">
          <TradingChart 
            symbol={selectedSymbol} 
            signals={signals}
            onSymbolChange={setSelectedSymbol}
          />
          
          {/* Signal Summary for Selected Symbol */}
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              📊 {selectedSymbol} Signal History
            </h3>
            <div className="space-y-4">
              {signals
                .filter(signal => signal.symbol === selectedSymbol)
                .slice(0, 5)
                .map((signal) => (
                  <SignalCard 
                    key={signal.id} 
                    signal={signal} 
                    showActions={true}
                    onFollowSignal={handleFollowSignal}
                    onWatchSignal={handleWatchSignal}
                  />
                ))}
              {signals.filter(signal => signal.symbol === selectedSymbol).length === 0 && (
                <div className="text-center py-8">
                  <div className="text-gray-400 mb-2">🔍</div>
                  <div className="text-sm text-gray-400">No signals found for {selectedSymbol}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {currentView === 'portfolio' && (
        <PortfolioDashboard />
      )}
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