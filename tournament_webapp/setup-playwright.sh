#!/bin/bash

echo "🎭 Setting up Playwright Testing for Tournament Webapp"
echo "=================================================="

# Navigate to tournament webapp directory
cd "$(dirname "$0")"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
npx playwright install

# Create test results directory
mkdir -p test-results
mkdir -p playwright-report

echo "✅ Playwright setup complete!"
echo ""
echo "🚀 Available test commands:"
echo "  npm test              - Run all tests"
echo "  npm run test:ui       - Run tests with UI mode"
echo "  npm run test:headed   - Run tests in headed mode (visible browser)"
echo "  npm run test:debug    - Debug tests"
echo "  npm run test:smoke    - Run smoke tests only"
echo "  npm run test:workflow - Run workflow tests only"
echo "  npm run test:api      - Run API integration tests only"
echo "  npm run test:report   - Show test report"
echo ""
echo "💡 Make sure both servers are running:"
echo "  Frontend: cd frontend && npm start (port 3000)"
echo "  Backend:  cd backend && python simple_server.py (port 10000)"
