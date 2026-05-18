from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes.health import router as health_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown


app = FastAPI(title="LangGraph Demo", version="1.0.0", lifespan=lifespan)

app.include_router(health_router)

