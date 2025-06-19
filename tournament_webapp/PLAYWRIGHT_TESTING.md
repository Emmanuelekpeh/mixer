# 🎭 Playwright E2E Testing for Tournament Webapp

## Setup

### 1. Install Playwright
```bash
# Windows
./setup-playwright.bat

# Linux/Mac  
./setup-playwright.sh

# Or manually:
npm install
npx playwright install
```

### 2. Start Both Servers
Make sure both servers are running before running tests:

```bash
# Terminal 1: Start Frontend (React)
cd frontend
npm start
# Should be running on http://localhost:3000

# Terminal 2: Start Backend (Python)
cd backend  
python simple_server.py
# Should be running on http://localhost:10000
```

## Running Tests

### Basic Commands
```bash
# Run all tests
npm test

# Run tests with visual UI
npm run test:ui

# Run tests in headed mode (see browser)
npm run test:headed

# Debug tests step by step
npm run test:debug
```

### Specific Test Suites
```bash
# Smoke tests (basic functionality)
npm run test:smoke

# Tournament workflow tests
npm run test:workflow

# API integration tests
npm run test:api

# Responsive & accessibility tests
npx playwright test tests/playwright/responsive-accessibility.spec.js
```

### Test Reports
```bash
# View HTML test report
npm run test:report

# Test results are also saved in:
# - test-results/
# - playwright-report/
```

## Test Suites

### 1. Smoke Tests (`smoke.spec.js`)
- ✅ Application loads without errors
- ✅ Basic navigation works
- ✅ API connectivity check
- ✅ React app initialization

### 2. Tournament Workflow (`tournament-workflow.spec.js`)
- 🏆 Complete tournament creation flow
- 📁 File upload functionality
- ⚙️ Tournament configuration
- 🎵 Audio playback in battles
- 🏅 Tournament results display
- 🗑️ Tournament deletion (tests missing feature)

### 3. API Integration (`api-integration.spec.js`)
- 🔌 Backend API connectivity
- 📊 Data fetching and persistence
- ❌ Error handling
- 🔄 Model loading verification

### 4. Responsive & Accessibility (`responsive-accessibility.spec.js`)
- 📱 Mobile device compatibility
- ♿ Accessibility features
- ⌨️ Keyboard navigation
- 🖱️ Touch interactions

## Browser Coverage
Tests run on:
- ✅ Chrome (Desktop)
- ✅ Firefox (Desktop)
- ✅ Safari (Desktop)
- ✅ Chrome Mobile
- ✅ Safari Mobile

## Configuration

The tests are configured in `playwright.config.js`:
- Frontend URL: `http://localhost:3000`
- Backend URL: `http://localhost:10000`
- Automatic server startup (if not running)
- Screenshot on failure
- Video recording on failure
- Test traces for debugging

## Debugging Failed Tests

1. **View Screenshots**: Check `test-results/` folder
2. **Watch Videos**: Recorded on test failures
3. **Use Debug Mode**: `npm run test:debug`
4. **Check Traces**: Use Playwright trace viewer
5. **View Reports**: `npm run test:report` for detailed HTML report

## Expected Issues to Catch

Based on your earlier logs, these tests should help identify:

1. **Model Path Issues**: Tests will verify model loading
2. **Tournament Deletion**: Tests will confirm delete functionality is missing
3. **API Routing**: Tests will check all API endpoints work correctly
4. **File Upload**: Tests will verify audio file processing
5. **Database Persistence**: Tests will check data saves correctly

## CI/CD Integration

To use in CI/CD:
```yaml
- name: Run E2E Tests
  run: |
    npm install
    npx playwright install --with-deps
    npm test
```

## Troubleshooting

### Common Issues:
1. **Servers not running**: Make sure both frontend (3000) and backend (10000) are up
2. **Port conflicts**: Check if ports 3000/10000 are available
3. **Browser installation**: Run `npx playwright install` if browsers fail
4. **Test timeouts**: Increase timeout in config for slower systems

### Debug Commands:
```bash
# Check server status
curl http://localhost:3000
curl http://localhost:10000/health

# Run single test with full output
npx playwright test tests/playwright/smoke.spec.js --reporter=list

# Generate and view trace
npx playwright test --trace=on
npx playwright show-trace trace.zip
```
