"""
ControlPlane.ai - Enterprise LLMOps Observability Gateway
Main Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.routes import router as proxy_router
from app.api.feedback import router as feedback_router
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="ControlPlane.ai API Gateway",
    version="1.0.0",
    description="Real-time LLMOps inspection layer observing performance, cost, and responsibility."
)

# Enable CORS for Streamlit / React frontend dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount core proxy routes
app.include_router(proxy_router)
app.include_router(feedback_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "ControlPlane.ai Gateway"}

# 4. START PROMETHEUS METRICS
# This tracks request rates, latency histograms, and creates the /metrics endpoint
Instrumentator().instrument(app).expose(app, include_in_schema=True)


if __name__ == "__main__":
    import uvicorn
    # Local execution entry
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    
