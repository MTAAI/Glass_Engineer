# Glass Expert AI — Start API Server
# Run this from the project root in VS Code terminal

# Activate virtual environment
.\venv\Scripts\activate

# Start the FastAPI server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# The API will be available at:
#   http://localhost:8080
#   http://localhost:8080/docs        (Swagger UI — interactive API docs)
#   http://localhost:8080/api/v1/health
