import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  handleRetry = () => {
    // Reset the error boundary
    this.setState({ hasError: false, error: null, errorInfo: null });
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI for when market data fails to load
      return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-red-900/20 border border-red-600 rounded-lg p-8 text-center">
            <div className="mb-6">
              <div className="text-6xl mb-4">⚠️</div>
              <h2 className="text-2xl font-bold text-red-400 mb-2">
                Market Data Error
              </h2>
              <p className="text-gray-300 mb-4">
                There was an error loading the market data. This could be due to:
              </p>
              <ul className="text-sm text-gray-400 text-left max-w-md mx-auto space-y-1">
                <li>• Network connectivity issues</li>
                <li>• Exchange API temporarily unavailable</li>
                <li>• Invalid market data format</li>
                <li>• Browser compatibility issues</li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <button
                onClick={this.handleRetry}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2 mx-auto"
              >
                <span>🔄</span>
                <span>Retry Loading</span>
              </button>
              
              <button
                onClick={() => window.location.reload()}
                className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 mx-auto"
              >
                <span>↻</span>
                <span>Refresh Page</span>
              </button>
            </div>
            
            {/* Development error details */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-6 text-left bg-gray-800 rounded-lg p-4">
                <summary className="cursor-pointer text-yellow-400 font-medium mb-2">
                  🔍 Developer Details (Click to expand)
                </summary>
                <div className="text-xs font-mono text-gray-300 space-y-2">
                  <div>
                    <strong className="text-red-400">Error:</strong>
                    <pre className="whitespace-pre-wrap mt-1 bg-gray-900 p-2 rounded">
                      {this.state.error && this.state.error.toString()}
                    </pre>
                  </div>
                  <div>
                    <strong className="text-red-400">Stack Trace:</strong>
                    <pre className="whitespace-pre-wrap mt-1 bg-gray-900 p-2 rounded">
                      {this.state.errorInfo && this.state.errorInfo.componentStack}
                    </pre>
                  </div>
                </div>
              </details>
            )}
          </div>
        </div>
      );
    }

    // If no error, render children normally
    return this.props.children;
  }
}

export default ErrorBoundary;