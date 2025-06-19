# Start both the frontend and backend servers
# This script starts the backend in one window and the frontend in another

# Start the backend server
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python backend/main.py"

# Wait a moment for the backend to initialize
Start-Sleep -Seconds 2

# Start the frontend server in a new window
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot/frontend'; npm start"

Write-Host "Started both backend and frontend servers. Check the opened windows for details."
