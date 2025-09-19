import { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Settings = ({ config, setConfig }) => {
  const [formData, setFormData] = useState({
    symbols: [],
    max_positions: 5,
    risk_per_trade: 0.02,
    min_confidence: 0.8,
    trading_mode: 'sandbox',
    update_interval: 30
  });
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  useEffect(() => {
    if (config) {
      setFormData({
        symbols: config.symbols || [],
        max_positions: config.max_positions || 5,
        risk_per_trade: config.risk_per_trade || 0.02,
        min_confidence: config.min_confidence || 0.8,
        trading_mode: config.trading_mode || 'sandbox',
        update_interval: config.update_interval || 30
      });
    }
  }, [config]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage({ type: '', text: '' });

    try {
      const response = await axios.post(`${API}/config`, formData);
      setConfig(formData);
      setMessage({ 
        type: 'success', 
        text: 'Configuration updated successfully! Changes will take effect within 30 seconds.' 
      });
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: 'Failed to update configuration. Please try again.' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymbolChange = (symbol) => {
    const newSymbols = formData.symbols.includes(symbol)
      ? formData.symbols.filter(s => s !== symbol)
      : [...formData.symbols, symbol];
    
    setFormData({ ...formData, symbols: newSymbols });
  };

  const availableSymbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'MATIC-USD', 'DOT-USD'];

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">⚙️ Trading Settings</h1>
        <p className="text-gray-400">Configure your Quantum Flow trading parameters</p>
      </div>

      {/* Status Message */}
      {message.text && (
        <div className={`mb-6 p-4 rounded-lg border ${
          message.type === 'success' 
            ? 'bg-green-900/20 border-green-600 text-green-100'
            : 'bg-red-900/20 border-red-600 text-red-100'
        }`}>
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Trading Mode */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">🔄 Trading Mode</h2>
          
          <div className="space-y-4">
            <label className="flex items-center space-x-3">
              <input
                type="radio"
                name="trading_mode"
                value="sandbox"
                checked={formData.trading_mode === 'sandbox'}
                onChange={(e) => setFormData({ ...formData, trading_mode: e.target.value })}
                className="w-4 h-4 text-blue-600"
              />
              <div>
                <div className="text-white font-medium">Sandbox Mode (Recommended)</div>
                <div className="text-sm text-gray-400">
                  Safe testing environment with simulated trading. No real money at risk.
                </div>
              </div>
            </label>
            
            <label className="flex items-center space-x-3">
              <input
                type="radio"
                name="trading_mode"
                value="live"
                checked={formData.trading_mode === 'live'}
                onChange={(e) => setFormData({ ...formData, trading_mode: e.target.value })}
                className="w-4 h-4 text-blue-600"
              />
              <div>
                <div className="text-white font-medium">Live Trading</div>
                <div className="text-sm text-red-400">
                  ⚠️ Real trading with actual funds. Use with caution!
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Symbol Selection */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">📈 Trading Symbols</h2>
          <p className="text-gray-400 mb-4">Select which cryptocurrencies to monitor for trading signals</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {availableSymbols.map((symbol) => (
              <label key={symbol} className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg cursor-pointer hover:bg-gray-600">
                <input
                  type="checkbox"
                  checked={formData.symbols.includes(symbol)}
                  onChange={() => handleSymbolChange(symbol)}
                  className="w-4 h-4 text-blue-600"
                />
                <div>
                  <div className="text-white font-medium">
                    {symbol.replace('-USD', '')}
                  </div>
                  <div className="text-xs text-gray-400">{symbol}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Risk Management */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">🛡️ Risk Management</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Maximum Positions
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={formData.max_positions}
                onChange={(e) => setFormData({ ...formData, max_positions: parseInt(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              />
              <p className="text-xs text-gray-400 mt-1">Maximum number of active signals</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Risk Per Trade (%)
              </label>
              <input
                type="number"
                min="0.5"
                max="10"
                step="0.1"
                value={formData.risk_per_trade * 100}
                onChange={(e) => setFormData({ ...formData, risk_per_trade: parseFloat(e.target.value) / 100 })}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              />
              <p className="text-xs text-gray-400 mt-1">Percentage of portfolio risked per trade</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Minimum Confidence (%)
              </label>
              <input
                type="number"
                min="50"
                max="95"
                step="5"
                value={formData.min_confidence * 100}
                onChange={(e) => setFormData({ ...formData, min_confidence: parseFloat(e.target.value) / 100 })}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              />
              <p className="text-xs text-gray-400 mt-1">Minimum AI confidence to trigger signals</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Update Interval (seconds)
              </label>
              <select
                value={formData.update_interval}
                onChange={(e) => setFormData({ ...formData, update_interval: parseInt(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value={15}>15 seconds (High frequency)</option>
                <option value={30}>30 seconds (Recommended)</option>
                <option value={60}>1 minute</option>
                <option value={120}>2 minutes</option>
              </select>
              <p className="text-xs text-gray-400 mt-1">How often to scan for new signals</p>
            </div>
          </div>
        </div>

        {/* Signal Quality */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">🎯 Signal Quality</h2>
          
          <div className="space-y-4">
            <div className="p-4 bg-gray-700 rounded-lg">
              <h3 className="font-medium text-white mb-2">Current Filters</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Min Confidence:</span>
                  <span className="text-white ml-2">{(formData.min_confidence * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-gray-400">Max Risk/Trade:</span>
                  <span className="text-white ml-2">{(formData.risk_per_trade * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-400">Max Positions:</span>
                  <span className="text-white ml-2">{formData.max_positions}</span>
                </div>
                <div>
                  <span className="text-gray-400">Scan Frequency:</span>
                  <span className="text-white ml-2">{formData.update_interval}s</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">📊 System Status</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-900/20 border border-green-700 rounded-lg text-center">
              <div className="text-green-400 font-semibold">Exchange Connection</div>
              <div className="text-sm text-green-300 mt-1">✅ Coinbase Connected</div>
            </div>
            
            <div className="p-4 bg-green-900/20 border border-green-700 rounded-lg text-center">
              <div className="text-green-400 font-semibold">AI Engine</div>
              <div className="text-sm text-green-300 mt-1">✅ Groq AI Active</div>
            </div>
            
            <div className="p-4 bg-blue-900/20 border border-blue-700 rounded-lg text-center">
              <div className="text-blue-400 font-semibold">Mode</div>
              <div className="text-sm text-blue-300 mt-1">
                🔒 {formData.trading_mode === 'sandbox' ? 'Sandbox Safe' : 'Live Trading'}
              </div>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-8 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Saving...</span>
              </>
            ) : (
              <>
                <span>💾</span>
                <span>Save Configuration</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Settings;