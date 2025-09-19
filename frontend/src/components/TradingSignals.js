import { useState, useMemo } from 'react';
import SignalCard from './SignalCard';

const TradingSignals = ({ signals, activeSignals }) => {
  const [filter, setFilter] = useState('all'); // all, active, closed
  const [sortBy, setSortBy] = useState('timestamp'); // timestamp, confidence, symbol
  const [sortOrder, setSortOrder] = useState('desc'); // asc, desc

  const filteredAndSortedSignals = useMemo(() => {
    let filtered = [...signals];

    // Apply filter
    if (filter === 'active') {
      filtered = filtered.filter(signal => signal.status === 'ACTIVE');
    } else if (filter === 'closed') {
      filtered = filtered.filter(signal => signal.status === 'CLOSED');
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aVal, bVal;

      switch (sortBy) {
        case 'confidence':
          aVal = a.confidence;
          bVal = b.confidence;
          break;
        case 'symbol':
          aVal = a.symbol;
          bVal = b.symbol;
          break;
        case 'timestamp':
        default:
          aVal = new Date(a.timestamp);
          bVal = new Date(b.timestamp);
      }

      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered;
  }, [signals, filter, sortBy, sortOrder]);

  const stats = useMemo(() => {
    return {
      total: signals.length,
      active: signals.filter(s => s.status === 'ACTIVE').length,
      closed: signals.filter(s => s.status === 'CLOSED').length,
      profitable: signals.filter(s => s.actual_return && s.actual_return > 0).length
    };
  }, [signals]);

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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">🚀 Trading Signals</h1>
        <p className="text-gray-400">All detected quantum flow patterns and trading opportunities</p>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
          <div className="text-2xl font-bold text-white">{stats.total}</div>
          <div className="text-sm text-gray-400">Total Signals</div>
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
          <div className="text-2xl font-bold text-green-400">{stats.active}</div>
          <div className="text-sm text-gray-400">Active Signals</div>
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
          <div className="text-2xl font-bold text-blue-400">{stats.closed}</div>
          <div className="text-sm text-gray-400">Closed Signals</div>
        </div>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
          <div className="text-2xl font-bold text-yellow-400">{stats.profitable}</div>
          <div className="text-sm text-gray-400">Profitable</div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 mb-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
          {/* Filter */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-300">Filter:</label>
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
            >
              <option value="all">All Signals</option>
              <option value="active">Active Only</option>
              <option value="closed">Closed Only</option>
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-300">Sort by:</label>
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
            >
              <option value="timestamp">Time</option>
              <option value="confidence">Confidence</option>
              <option value="symbol">Symbol</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm hover:bg-gray-600"
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>
        </div>
      </div>

      {/* Signals List */}
      <div className="space-y-6">
        {filteredAndSortedSignals.length > 0 ? (
          filteredAndSortedSignals.map((signal) => (
            <SignalCard key={signal.id} signal={signal} showActions={true} />
          ))
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-12 text-center">
            <div className="text-6xl mb-4">🔍</div>
            <h3 className="text-xl font-semibold text-white mb-2">No Signals Found</h3>
            <p className="text-gray-400">
              {filter === 'all' 
                ? "No signals have been detected yet. The system is monitoring markets..."
                : `No ${filter} signals match your current filter.`
              }
            </p>
            {filter !== 'all' && (
              <button
                onClick={() => setFilter('all')}
                className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm font-medium"
              >
                Show All Signals
              </button>
            )}
          </div>
        )}
      </div>

      {/* Load More (if needed) */}
      {filteredAndSortedSignals.length >= 50 && (
        <div className="text-center mt-8">
          <button className="bg-gray-700 hover:bg-gray-600 text-white px-6 py-2 rounded text-sm font-medium">
            Load More Signals
          </button>
        </div>
      )}
    </div>
  );
};

export default TradingSignals;