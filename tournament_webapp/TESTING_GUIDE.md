# Tournament Web App Testing Guide

This guide will help you test the Tournament Web App using Playwright, including testing the tournament deletion functionality which has been fixed.

## Prerequisites

- Node.js (version 14 or higher)
- npm (comes with Node.js)
- Python 3.8 or higher

## Setup

1. First, install the dependencies for both the backend and frontend:

```powershell
# Install backend dependencies
cd tournament_webapp/backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install

# Install Playwright dependencies
cd ..
npm install
npx playwright install
```

## Running the Tests

1. Start the backend server:

```powershell
cd tournament_webapp/backend
python simple_server.py
```

2. In a separate terminal, start the frontend server:

```powershell
cd tournament_webapp/frontend
npm start
```

3. In a third terminal, run the Playwright tests:

```powershell
cd tournament_webapp
npx playwright test
```

4. To run specific tests:

```powershell
# Run only the tournament deletion tests
npx playwright test tournament-deletion.spec.js

# Run with visible browser (for debugging)
npx playwright test --headed

# Run and show HTML report afterward
npx playwright test --reporter=html && npx playwright show-report
```

## Viewing Test Results

After running the tests, you can view the HTML report:

```powershell
npx playwright show-report
```

## Key Issues Fixed

1. **Tournament Deletion**: The frontend now correctly calls the API to delete tournaments and refreshes the list afterward
2. **Model Path Resolution**: Fixed the models directory path resolution to use absolute paths
3. **Model Files Endpoint**: Added a new `/api/model-files` endpoint to get detailed information about model files for debugging
4. **Improved Error Handling**: Better error handling in the frontend for tournament deletion

## Troubleshooting

If you encounter any issues:

1. Make sure both backend and frontend servers are running
2. Check the browser console for errors
3. Check the terminal running the backend server for Python errors
4. Try running individual tests to isolate problems

For specific deletion issues, check the network tab in your browser's developer tools to see if the DELETE request is being sent correctly.
