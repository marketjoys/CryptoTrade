import { useState, useEffect, useRef } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import "./App.css";

// Import components (we'll create these)
import Dashboard from "./components/Dashboard";
import TradingSignals from "./components/TradingSignals";
import Performance from "./components/Performance";
import MarketData from "./components/MarketData";
import Settings from "./components/Settings";
import Navbar from "./components/Navbar";
import ErrorBoundary from "./components/ErrorBoundary";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [signals, setSignals] = useState([]);
  const [activeSignals, setActiveSignals] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [marketData, setMarketData] = useState({});
  const [config, setConfig] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://');
      wsRef.current = new WebSocket(`${wsUrl}/api/ws`);
      
      wsRef.current.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'new_signal') {
            setSignals(prev => [message.data, ...prev]);
            setActiveSignals(prev => [...prev, message.data]);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setIsConnected(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    fetchInitialData();
    
    // Set up polling for data that doesn't come via WebSocket
    const interval = setInterval(() => {
      fetchPerformance();
      fetchActiveSignals();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchInitialData = async () => {
    try {
      const [signalsRes, activeSignalsRes, performanceRes, configRes] = await Promise.all([
        axios.get(`${API}/signals`),
        axios.get(`${API}/signals/active`),
        axios.get(`${API}/performance`),
        axios.get(`${API}/config`)
      ]);

      setSignals(signalsRes.data || []);
      setActiveSignals(activeSignalsRes.data || []);
      setPerformance(performanceRes.data);
      setConfig(configRes.data);
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await axios.get(`${API}/performance`);
      setPerformance(response.data);
    } catch (error) {
      console.error('Error fetching performance:', error);
    }
  };

  const fetchActiveSignals = async () => {
    try {
      const response = await axios.get(`${API}/signals/active`);
      setActiveSignals(response.data || []);
    } catch (error) {
      console.error('Error fetching active signals:', error);
    }
  };

  const fetchMarketData = async (symbol) => {
    try {
      const response = await axios.get(`${API}/market-data/${symbol}`);
      setMarketData(prev => ({
        ...prev,
        [symbol]: response.data
      }));
    } catch (error) {
      console.error(`Error fetching market data for ${symbol}:`, error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <BrowserRouter>
        <Navbar isConnected={isConnected} />
        
        <main className="pt-16">
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  signals={signals}
                  activeSignals={activeSignals}
                  performance={performance}
                  marketData={marketData}
                  isConnected={isConnected}
                />
              } 
            />
            <Route 
              path="/signals" 
              element={
                <TradingSignals 
                  signals={signals}
                  activeSignals={activeSignals}
                />
              } 
            />
            <Route 
              path="/performance" 
              element={
                <Performance 
                  performance={performance}
                  signals={signals}
                />
              } 
            />
            <Route 
              path="/market" 
              element={
                <MarketData 
                  marketData={marketData}
                  fetchMarketData={fetchMarketData}
                  config={config}
                />
              } 
            />
            <Route 
              path="/settings" 
              element={
                <Settings 
                  config={config}
                  setConfig={setConfig}
                />
              } 
            />
          </Routes>
        </main>
      </BrowserRouter>
    </div>
  );
}

export default App;