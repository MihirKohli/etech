from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes.health import router as health_router
from db.sql_database import init_db
from db.vector_database import get_collection, get_embeddings

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    get_embeddings()
    get_collection()
    # startup
    yield
    # shutdown


app = FastAPI(title="LangGraph Demo", version="1.0.0", lifespan=lifespan)

app.include_router(health_router)

