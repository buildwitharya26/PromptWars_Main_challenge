import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

from config import settings, logger
from agents import TravelOrchestratorAgent, TravelPlanRequest
from google import genai

# ====================================================
# PYDANTIC INPUT REQUEST MODELS
# ====================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Chat message from the user")
    session_id: str = Field(default="", max_length=100, description="Session ID for conversation memory")
    destination: str = Field(default="", max_length=100, description="Optional target destination")

class WeatherRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=100, description="Target destination for weather forecast")

# ====================================================
# LIFESPAN & DEPENDENCY INJECTION
# ====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes Google GenAI Client and Orchestrator Agent on startup."""
    logger.info("WanderAI starting up...")
    try:
        # Initialize the new Google GenAI Client
        # Using settings configured via environment variables
        client = genai.Client(
            vertexai=settings.GOOGLE_GENAI_USE_VERTEXAI,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.GOOGLE_CLOUD_LOCATION
        )
        app.state.orchestrator = TravelOrchestratorAgent(client)
        logger.info("Google GenAI client and TravelOrchestrator initialized successfully")
    except Exception as e:
        logger.exception(f"Startup error initializing Google GenAI Client: {e}")
        # Allow fallback initialization for local testing/pytest mock scenarios
        app.state.orchestrator = None
        
    yield
    logger.info("WanderAI shutting down...")


app = FastAPI(
    title="WanderAI API",
    description="Production-grade backend for WanderAI - Your AI Travel Planner",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None
)

# ====================================================
# MIDDLEWARES
# ====================================================

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to actual domains in final production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request Size Limit Middleware (1MB limit)
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.method in ("POST", "PUT"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > 1024 * 1024:  # 1 MB
                    logger.warning(f"Payload too large: {length} bytes")
                    return HTMLResponse(
                        content="Payload Too Large", 
                        status_code=status.HTTP_413_PAYLOAD_TOO_LARGE
                    )
            except ValueError:
                return HTMLResponse(
                    content="Invalid Content-Length Header", 
                    status_code=status.HTTP_400_BAD_REQUEST
                )
    return await call_next(request)

# Secure Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://images.unsplash.com; "
        "connect-src 'self';"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Setup templates
templates = Jinja2Templates(directory="templates")

# Dependency Injection for Orchestrator
def get_orchestrator(request: Request) -> TravelOrchestratorAgent:
    orchestrator = request.app.state.orchestrator
    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI planning service is currently unavailable. Please verify API configuration."
        )
    return orchestrator

# ====================================================
# FASTAPI ENDPOINTS
# ====================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main single-page application interface."""
    logger.info("Serving root landing page")
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
async def health_check():
    """Simple health check endpoint for Cloud Run/Kubernetes."""
    logger.debug("Health check requested")
    return {
        "status": "ok",
        "service": "WanderAI"
    }


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    orchestrator: TravelOrchestratorAgent = Depends(get_orchestrator)
):
    """Orchestrates multi-turn chat conversations with memory & context."""
    # Ensure session_id exists
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Chat request received for session: {session_id}")
    
    try:
        response_data = await orchestrator.chat(
            message=request.message,
            session_id=session_id,
            destination=request.destination
        )
        return response_data
    except Exception as e:
        logger.exception(f"Unhandled error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your chat query."
        )


@app.post("/weather")
async def weather_endpoint(
    request: WeatherRequest,
    orchestrator: TravelOrchestratorAgent = Depends(get_orchestrator)
):
    """Direct weather lookup endpoint invoking WeatherAgent."""
    logger.info(f"Weather lookup request for: {request.destination}")
    try:
        weather_details = await orchestrator.weather_agent.generate(request.destination)
        return weather_details.model_dump()
    except Exception as e:
        logger.exception(f"Error in weather endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve weather for {request.destination}."
        )


@app.post("/itinerary")
async def itinerary_endpoint(
    request: TravelPlanRequest,
    orchestrator: TravelOrchestratorAgent = Depends(get_orchestrator)
):
    """Direct itinerary generation endpoint invoking ItineraryAgent."""
    logger.info(f"Itinerary generation request for: {request.destination} ({request.days} days)")
    try:
        itinerary_details = await orchestrator.itinerary_agent.generate(
            destination=request.destination,
            days=request.days,
            budget=request.budget,
            interests=request.interests,
            travelers=request.travelers
        )
        return itinerary_details.model_dump()
    except Exception as e:
        logger.exception(f"Error in itinerary endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate itinerary for {request.destination}."
        )
