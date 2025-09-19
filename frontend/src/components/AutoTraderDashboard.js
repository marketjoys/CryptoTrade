import React, { useState, useEffect } from 'react';

const AutoTraderDashboard = () => {
  const [autoTraderStatus, setAutoTraderStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  useEffect(() => {
    fetchAutoTraderStatus();
    // Refresh status every 10 seconds
    const interval = setInterval(fetchAutoTraderStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchAutoTraderStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/autotrader/status`);
      const data = await response.json();
      setAutoTraderStatus(data);
    } catch (error) {
      console.error('Error fetching AutoTrader status:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleAutoTrader = async () => {
    setUpdating(true);
    try {
      const endpoint = autoTraderStatus.enabled ? 'disable' : 'enable';
      const response = await fetch(`${backendUrl}/api/autotrader/${endpoint}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        await fetchAutoTraderStatus();
        const action = autoTraderStatus.enabled ? 'disabled' : 'enabled';
        alert(`AutoTrader ${action} successfully!`);
      } else {
        throw new Error('Failed to toggle AutoTrader');
      }
    } catch (error) {
      console.error('Error toggling AutoTrader:', error);
      alert('Failed to toggle AutoTrader');
    } finally {
      setUpdating(false);
    }
  };

  const updateConfig = async (newConfig) => {
    setUpdating(true);
    try {
      const response = await fetch(`${backendUrl}/api/autotrader/config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newConfig),
      });
      
      if (response.ok) {
        await fetchAutoTraderStatus();
        alert('AutoTrader configuration updated successfully!');
      } else {
        throw new Error('Failed to update configuration');
      }
    } catch (error) {
      console.error('Error updating config:', error);
      alert('Failed to update configuration');
    } finally {
      setUpdating(false);
    }
  };

  const formatPercentage = (value) => `${(value * 100).toFixed(1)}%`;
  const formatCurrency = (value) => `$${value.toFixed(2)}`;

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-white">Loading AutoTrader...</span>
        </div>
      </div>
    );
  }

  if (!autoTraderStatus) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="text-red-400 text-center">
          ❌ Failed to load AutoTrader status
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="text-2xl">🤖</div>
            <div>
              <h2 className="text-xl font-bold text-white">AutoTrader</h2>
              <p className="text-gray-400 text-sm">Automated high-confidence trading</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              autoTraderStatus.enabled 
                ? 'bg-green-900 text-green-300' 
                : 'bg-red-900 text-red-300'
            }`}>
              {autoTraderStatus.enabled ? '🟢 ACTIVE' : '🔴 INACTIVE'}
            </div>
            
            <button
              onClick={toggleAutoTrader}
              disabled={updating}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                updating
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : autoTraderStatus.enabled
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {updating ? '⏳' : autoTraderStatus.enabled ? 'DISABLE' : 'ENABLE'}
            </button>
          </div>
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-blue-400 text-sm font-medium">Active Positions</div>
          <div className="text-2xl font-bold text-white mt-1">
            {autoTraderStatus.active_positions}
          </div>
          <div className="text-gray-400 text-xs mt-1">
            Max: {autoTraderStatus.config.max_positions}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-green-400 text-sm font-medium">Portfolio Risk</div>
          <div className="text-2xl font-bold text-white mt-1">
            {formatPercentage(autoTraderStatus.total_risk)}
          </div>
          <div className="text-gray-400 text-xs mt-1">
            Max: {formatPercentage(autoTraderStatus.config.max_portfolio_risk)}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-purple-400 text-sm font-medium">Min Confidence</div>
          <div className="text-2xl font-bold text-white mt-1">
            {formatPercentage(autoTraderStatus.config.min_confidence)}
          </div>
          <div className="text-gray-400 text-xs mt-1">
            High threshold
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="text-yellow-400 text-sm font-medium">Risk Per Trade</div>
          <div className="text-2xl font-bold text-white mt-1">
            {formatPercentage(autoTraderStatus.config.risk_per_trade)}
          </div>
          <div className="text-gray-400 text-xs mt-1">
            Conservative
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-bold text-white mb-4">⚙️ Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Minimum Confidence Threshold
              </label>
              <div className="text-white">
                {formatPercentage(autoTraderStatus.config.min_confidence)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Only signals above this confidence will be auto-traded
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Max Positions
              </label>
              <div className="text-white">
                {autoTraderStatus.config.max_positions}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Maximum concurrent auto-trading positions
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Risk Per Trade
              </label>
              <div className="text-white">
                {formatPercentage(autoTraderStatus.config.risk_per_trade)}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Percentage of portfolio risked per trade
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Profit Target Multiplier
              </label>
              <div className="text-white">
                {autoTraderStatus.config.profit_target_multiplier.toFixed(2)}x
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Minimum expected profit multiplier
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Stop Loss Multiplier
              </label>
              <div className="text-white">
                {autoTraderStatus.config.stop_loss_multiplier.toFixed(2)}x
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Automatic stop loss level
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Cooldown Period
              </label>
              <div className="text-white">
                {autoTraderStatus.config.cooldown_minutes} minutes
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Wait time between trades on same symbol
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Active Positions */}
      {autoTraderStatus.positions.length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">📊 Active Auto-Positions</h3>
          
          <div className="space-y-3">
            {autoTraderStatus.positions.map((position, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="text-lg font-bold text-white">
                      {position.symbol}
                    </div>
                    <div className="px-2 py-1 bg-blue-900 text-blue-300 rounded text-sm">
                      {position.flow_type}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-white font-medium">
                      Entry: {formatCurrency(position.entry_price)}
                    </div>
                    <div className="text-green-400 text-sm">
                      Target: {formatCurrency(position.target_price)}
                    </div>
                    <div className="text-red-400 text-sm">
                      Stop: {formatCurrency(position.stop_loss)}
                    </div>
                  </div>
                </div>
                
                <div className="mt-2 text-xs text-gray-400">
                  Risk: {formatCurrency(position.risk_amount)} | 
                  Entry Time: {new Date(position.entry_time).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      {Object.keys(autoTraderStatus.last_trades).length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4">🕒 Recent Auto-Trades</h3>
          
          <div className="space-y-2">
            {Object.entries(autoTraderStatus.last_trades).map(([symbol, timestamp]) => (
              <div key={symbol} className="flex items-center justify-between text-sm">
                <div className="text-white font-medium">{symbol}</div>
                <div className="text-gray-400">
                  {new Date(timestamp).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AutoTraderDashboard;