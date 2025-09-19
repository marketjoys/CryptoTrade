import { useState } from 'react';

const SignalCard = ({ signal, showActions = false }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getFlowTypeColor = (flowType) => {
    const colors = {
      'WHALE_ACCUMULATION': 'bg-blue-600',
      'LIQUIDITY_VACUUM': 'bg-purple-600',
      'MOMENTUM_SPIRAL': 'bg-green-600',
      'VOLUME_ANOMALY': 'bg-yellow-600'
    };
    return colors[flowType] || 'bg-gray-600';
  };

  const getFlowTypeIcon = (flowType) => {
    const icons = {
      'WHALE_ACCUMULATION': '🐋',
      'LIQUIDITY_VACUUM': '💫',
      'MOMENTUM_SPIRAL': '🌪️',
      'VOLUME_ANOMALY': '📊'
    };
    return icons[flowType] || '📈';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-400';
    if (confidence >= 0.8) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const calculatePotentialGain = () => {
    return ((signal.target_multiplier - 1) * 100).toFixed(1);
  };

  return (
    <div className="bg-gray-700 rounded-lg border border-gray-600 p-4 hover:border-gray-500 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className={`w-10 h-10 ${getFlowTypeColor(signal.flow_type)} rounded-lg flex items-center justify-center text-white text-lg`}>
            {getFlowTypeIcon(signal.flow_type)}
          </div>
          <div>
            <h3 className="font-semibold text-white">{signal.symbol}</h3>
            <p className="text-sm text-gray-400">{signal.flow_type.replace('_', ' ')}</p>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`text-lg font-bold ${getConfidenceColor(signal.confidence)}`}>
            {(signal.confidence * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Confidence</div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-sm font-medium text-white">
            {formatPrice(signal.entry_price)}
          </div>
          <div className="text-xs text-gray-400">Entry Price</div>
        </div>
        <div className="text-center">
          <div className="text-sm font-medium text-green-400">
            +{calculatePotentialGain()}%
          </div>
          <div className="text-xs text-gray-400">Target</div>
        </div>
        <div className="text-center">
          <div className="text-sm font-medium text-red-400">
            {(signal.risk_factor * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400">Risk</div>
        </div>
      </div>

      {/* Status and AI Conviction */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className={`px-2 py-1 rounded text-xs font-medium ${
            signal.status === 'ACTIVE' 
              ? 'bg-green-600 text-green-100' 
              : 'bg-gray-600 text-gray-100'
          }`}>
            {signal.status}
          </span>
          <span className="text-xs text-gray-400">
            AI: {signal.ai_conviction}
          </span>
        </div>
        
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-blue-400 hover:text-blue-300"
        >
          {isExpanded ? 'Less' : 'More'} Details
        </button>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-gray-600 pt-3 mt-3">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-gray-400">Flow Strength:</span>
              <span className="text-white ml-1">{signal.flow_strength.toFixed(3)}</span>
            </div>
            <div>
              <span className="text-gray-400">Network Effect:</span>
              <span className="text-white ml-1">{signal.network_effect.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-gray-400">Exchange:</span>
              <span className="text-white ml-1">{signal.exchange}</span>
            </div>
            <div>
              <span className="text-gray-400">Created:</span>
              <span className="text-white ml-1">{formatTime(signal.timestamp)}</span>
            </div>
          </div>
          
          {/* Exit Strategy */}
          <div className="mt-3 p-2 bg-gray-800 rounded text-xs">
            <div className="font-medium text-gray-300 mb-1">Exit Strategy:</div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <span className="text-gray-400">Profit Target:</span>
                <span className="text-green-400 ml-1">
                  {(signal.exit_strategy.profit_target * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Stop Loss:</span>
                <span className="text-red-400 ml-1">
                  -{(signal.exit_strategy.stop_loss * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Time Limit:</span>
                <span className="text-white ml-1">
                  {signal.exit_strategy.time_limit}h
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {showActions && signal.status === 'ACTIVE' && (
        <div className="border-t border-gray-600 pt-3 mt-3">
          <div className="flex space-x-2">
            <button className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded text-sm font-medium transition-colors">
              🚀 Follow Signal
            </button>
            <button className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded text-sm font-medium transition-colors">
              👁️ Watch
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalCard;