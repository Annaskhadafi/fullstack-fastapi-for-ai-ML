import os
import logging
import asyncio
from contextlib import suppress
from app.services.market_forecast_service import monitor_forecasts
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.services.auth_service import seed_default_admin
from app.services.ml_service import seed_demo_models
from app.api.v1.router import api_v1_router
from app.web.auth_views import router as web_auth_router
from app.web.dashboard_views import router as web_dashboard_router
from app.web.cv_views import router as web_cv_router
from app.web.rag_views import router as web_rag_router
from app.web.rag_vectorless_views import router as web_rag_vectorless_router
from app.web.ml_views import router as web_ml_router
from app.web.models_views import router as web_models_router
from app.web.face_views import router as web_face_router
from app.web.admin_views import router as web_admin_router
from app.web.forecast_views import router as web_forecast_router

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    logger.info("Initializing FastAPI AI Monolith...")

    # Ensure weights and static directories exist
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    os.makedirs("app/static", exist_ok=True)

    # Initialize Database Schema & Extensions
    try:
        await init_db()
        logger.info(f"Connected to database ({'PostgreSQL/Neon' if settings.is_postgres else 'SQLite Local'})")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    # Seed Initial Admin / First User
    try:
        async with AsyncSessionLocal() as session:
            await seed_default_admin(session)
    except Exception as e:
        logger.warning(f"Could not seed initial admin user on startup: {e}")

    # Seed Demo ML Models
    try:
        async with AsyncSessionLocal() as session:
            await seed_demo_models(session)
    except Exception as e:
        logger.warning(f"Could not seed demo models on startup: {e}")

    logger.info(f"{settings.APP_NAME} is ready to serve traffic!")
    forecast_task = asyncio.create_task(monitor_forecasts())
    try:
        yield
    finally:
        forecast_task.cancel()
        with suppress(asyncio.CancelledError):
            await forecast_task
    logger.info("Shutting down application...")


# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="Monolith boilerplate for Computer Vision (OpenCV + YOLO ONNX), RAG (Neon pgvector), and Machine Learning (Scikit-Learn/ONNX).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware (allows programmatic API consumers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include REST API Routers
app.include_router(api_v1_router)

# Include Monolith Web UI Routers
app.include_router(web_auth_router)
app.include_router(web_dashboard_router)
app.include_router(web_cv_router)
app.include_router(web_rag_router)
app.include_router(web_rag_vectorless_router)
app.include_router(web_ml_router)
app.include_router(web_models_router)
app.include_router(web_face_router)
app.include_router(web_admin_router)
app.include_router(web_forecast_router)


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Healthcheck endpoint for Docker, Dokploy, and Vercel."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database": "postgresql (Neon DB)" if settings.is_postgres else "sqlite",
        "environment": settings.APP_ENV
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
