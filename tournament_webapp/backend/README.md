# AI Mixer Tournament Backend

This is the backend server for the AI Mixer Tournament application. It provides the API endpoints for creating tournaments, managing users, and processing battle votes.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository (if you haven't already)
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r backend/requirements.txt
```

### Running the Server

To start the backend server:

```bash
python backend/main.py
```

The server will start on http://localhost:8000 by default.

## API Endpoints

### User Management

- `POST /api/users` - Create a new user
  - Request body: `{ "name": "User Name", "email": "optional@email.com" }`
  - Response: `{ "success": true, "user": { ... } }`

### Tournament Management

- `POST /api/tournaments` - Create a new tournament
  - Request body: `{ "user_id": "user_123", "max_rounds": 5, "audio_file": "optional.mp3" }`
  - Response: `{ "success": true, "tournament": { ... } }`

- `GET /api/tournaments/{tournament_id}` - Get tournament details
  - Response: `{ "success": true, "tournament": { ... } }`

- `POST /api/tournaments/{tournament_id}/vote` - Submit a vote for a model in a battle
  - Request body: `{ "model_id": "model_123", "confidence": 0.8 }`
  - Response: `{ "success": true, "battle_result": { ... }, "next_pair": { ... } }`

## Development

The backend is built with FastAPI and uses a simplified tournament engine for managing model battles.

### Directory Structure

- `backend/` - Backend code
  - `main.py` - Entry point for the server
  - `tournament_api.py` - API endpoint definitions
  - `simplified_tournament_engine.py` - Tournament logic implementation

### Testing

To run tests:

```bash
pytest
```

## Integration with Frontend

The frontend application is configured to use these API endpoints. Make sure the backend server is running before using the frontend application.

## Environment Variables

- `PORT` - Port for the server (default: 8000)
- `HOST` - Host address (default: 0.0.0.0)
- `ALLOWED_ORIGINS` - Comma-separated list of allowed CORS origins (default: http://localhost:3000,http://localhost:8080)
- `API_PREFIX` - Prefix for all API routes (default: "")
- `DATA_DIR` - Directory for storing data (default: "data")
