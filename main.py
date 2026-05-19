from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes.health import router as health_router
from db.sql_database import init_db
from routes.session import router as session_router
from routes.chat import router as chat_router
from routes.document import router as document_router
from routes.memories import router as memory_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # startup
    yield
    # shutdown


app = FastAPI(title="LangGraph Demo", version="1.0.0", lifespan=lifespan)

app.include_router(chat_router)
app.include_router(document_router)
app.include_router(health_router)
app.include_router(memory_router)
app.include_router(session_router)

