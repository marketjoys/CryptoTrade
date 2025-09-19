import { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const GroqStats = () => {
  const [groqStats, setGroqStats] = useState({
    total_calls: 0,
    last_call_time: null,
    calls_per_minute: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchGroqStats();
    
    // Update stats every 30 seconds
    const interval = setInterval(fetchGroqStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchGroqStats = async () => {
    try {
      const response = await axios.get(`${API}/groq-stats`);
      setGroqStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching Groq stats:', error);
      setLoading(false);
    }
  };

  const formatLastCallTime = (timestamp) => {
    if (!timestamp) return 'Never';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">🤖 AI Analysis</h3>
        <div className="flex items-center justify-center py-8">
          <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">🤖 AI Analysis</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-gray-400">Groq API</span>
        </div>
      </div>
      
      <div className="space-y-4">
        {/* Total API Calls */}
        <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
              📊
            </div>
            <div>
              <div className="font-medium text-white">Total AI Calls</div>
              <div className="text-xs text-gray-400">Since system start</div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-white">{groqStats.total_calls}</div>
          </div>
        </div>

        {/* API Call Frequency */}
        <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg flex items-center justify-center">
              ⚡
            </div>
            <div>
              <div className="font-medium text-white">Call Frequency</div>
              <div className="text-xs text-gray-400">Calls per minute</div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-white">
              {groqStats.calls_per_minute.toFixed(1)}
            </div>
          </div>
        </div>

        {/* Last Call Time */}
        <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
              🕒
            </div>
            <div>
              <div className="font-medium text-white">Last AI Call</div>
              <div className="text-xs text-gray-400">Most recent analysis</div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-white">
              {formatLastCallTime(groqStats.last_call_time)}
            </div>
          </div>
        </div>
      </div>

      {/* Status Indicator */}
      <div className="mt-4 pt-3 border-t border-gray-600">
        <div className="flex items-center justify-center space-x-2 text-xs text-gray-400">
          <div className={`w-1.5 h-1.5 rounded-full ${
            groqStats.total_calls > 0 ? 'bg-green-400' : 'bg-yellow-400'
          }`}></div>
          <span>
            {groqStats.total_calls > 0 
              ? 'AI analysis active' 
              : 'Waiting for signals...'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default GroqStats;