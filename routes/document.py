import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from services.document_ingestion import ingest_document
from db.sql_database import get_db
from services.session_management import get_user_memories
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
 
UPLOAD_DIR = "./data/uploads"
 


# ── Document Upload ──────────────────────────────────
 
@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    allowed = {".pdf", ".md", ".html", ".htm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")
 
    # Save to disk
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
 
    # Ingest into vector store
    result = ingest_document(filepath)
 
    return {
        "document_id": result["document_id"],
        "filename": result["filename"],
        "chunks_created": result["chunks_created"],
    }
 
 
# ── Memories ─────────────────────────────────────────
 
@router.get("/memories/{user_id}")
async def get_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    memories = await get_user_memories(db, user_id)
    return [
        {
            "memory_type": m.memory_type,
            "content": m.content,
            "importance": m.importance,
            "created_at": m.created_at,
        }
        for m in memories
    ]
