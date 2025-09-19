import React, { useState, useEffect } from 'react';

const PortfolioDashboard = () => {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);

  useEffect(() => {
    loadPortfolio();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadPortfolio, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadPortfolio = async () => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/portfolio`);
      const data = await response.json();
      setPortfolio(data);
    } catch (error) {
      console.error('Error loading portfolio:', error);
    } finally {
      setLoading(false);
    }
  };

  const updatePositions = async () => {
    setUpdating(true);
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/portfolio/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const result = await response.json();
      if (result.portfolio) {
        setPortfolio(result.portfolio);
      }
    } catch (error) {
      console.error('Error updating portfolio:', error);
    } finally {
      setUpdating(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount || 0);
  };

  const formatPercentage = (percentage) => {
    return `${(percentage || 0).toFixed(2)}%`;
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getPositionStatusColor = (status) => {
    const colors = {
      'ACTIVE': 'bg-green-600 text-green-100',
      'CLOSED': 'bg-gray-600 text-gray-100',
      'WATCHING': 'bg-blue-600 text-blue-100'
    };
    return colors[status] || 'bg-gray-600 text-gray-100';
  };

  const getPnLColor = (pnl) => {
    if (pnl > 0) return 'text-green-400';
    if (pnl < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-4"></div>
        <div className="text-white">Loading portfolio...</div>
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <div className="text-red-400 mb-2">❌ Error loading portfolio</div>
        <button 
          onClick={loadPortfolio}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-white">💼 Mock Trading Portfolio</h2>
          <button
            onClick={updatePositions}
            disabled={updating}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              updating 
                ? 'bg-gray-600 cursor-not-allowed text-gray-300' 
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {updating ? '⏳ Updating...' : '🔄 Update Positions'}
          </button>
        </div>

        {/* Portfolio Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {formatCurrency(portfolio.current_balance)}
            </div>
            <div className="text-sm text-gray-400">Current Balance</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className={`text-2xl font-bold ${getPnLColor(portfolio.total_pnl)}`}>
              {formatCurrency(portfolio.total_pnl)}
            </div>
            <div className="text-sm text-gray-400">Total P&L</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className={`text-2xl font-bold ${getPnLColor(portfolio.total_pnl_percentage)}`}>
              {formatPercentage(portfolio.total_pnl_percentage)}
            </div>
            <div className="text-sm text-gray-400">Total Return</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {formatPercentage(portfolio.win_rate * 100)}
            </div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-lg font-medium text-green-400">
              {portfolio.active_positions}
            </div>
            <div className="text-xs text-gray-400">Active Positions</div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-medium text-blue-400">
              {portfolio.closed_positions}
            </div>
            <div className="text-xs text-gray-400">Closed Positions</div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-medium text-white">
              {formatCurrency(portfolio.initial_balance)}
            </div>
            <div className="text-xs text-gray-400">Starting Capital</div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-medium text-purple-400">
              {portfolio.positions?.filter(p => p.status === 'WATCHING').length || 0}
            </div>
            <div className="text-xs text-gray-400">Watching</div>
          </div>
        </div>
      </div>

      {/* Positions List */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-xl font-bold text-white mb-4">📊 Positions</h3>
        
        {portfolio.positions && portfolio.positions.length > 0 ? (
          <div className="space-y-4">
            {portfolio.positions.map((position) => (
              <div key={position.id} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <div className="text-lg font-medium text-white">
                      {position.symbol}
                    </div>
                    <div className="text-sm text-gray-400">
                      {position.flow_type.replace('_', ' ')}
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getPositionStatusColor(position.status)}`}>
                      {position.status}
                    </span>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Entry Time</div>
                    <div className="text-xs text-white">{formatTime(position.entry_time)}</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">Entry Price</div>
                    <div className="text-white font-medium">
                      {formatCurrency(position.entry_price)}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-gray-400">Quantity</div>
                    <div className="text-white font-medium">
                      {position.quantity.toFixed(6)}
                    </div>
                  </div>
                  
                  {position.status === 'ACTIVE' && position.quantity > 0 && (
                    <>
                      <div>
                        <div className="text-gray-400">P&L</div>
                        <div className={`font-medium ${getPnLColor(position.pnl)}`}>
                          {formatCurrency(position.pnl || 0)}
                        </div>
                      </div>
                      
                      <div>
                        <div className="text-gray-400">P&L %</div>
                        <div className={`font-medium ${getPnLColor(position.pnl_percentage)}`}>
                          {formatPercentage(position.pnl_percentage || 0)}
                        </div>
                      </div>
                    </>
                  )}
                  
                  {position.status === 'WATCHING' && (
                    <div className="col-span-2">
                      <div className="text-blue-400 text-sm">👁️ Watching - No financial position</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-6xl mb-4">📈</div>
            <h3 className="text-xl font-semibold text-white mb-2">No Positions Yet</h3>
            <p className="text-gray-400">
              Start following signals to build your mock trading portfolio
            </p>
          </div>
        )}
      </div>

      {/* Portfolio Performance Chart Placeholder */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-xl font-bold text-white mb-4">📈 Performance History</h3>
        <div className="text-center py-8 text-gray-400">
          <div className="text-4xl mb-2">📊</div>
          <p>Performance chart coming soon...</p>
          <p className="text-sm mt-2">Track your portfolio performance over time</p>
        </div>
      </div>
    </div>
  );
};

export default PortfolioDashboard;