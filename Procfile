web: uvicorn tournament_webapp.backend.main:app --host 0.0.0.0 --port $PORT
worker: python tournament_webapp/backend/async_task_manager.py