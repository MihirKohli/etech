import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.document_ingestion import ingest_document

router = APIRouter()
 
UPLOAD_DIR = "./data/uploads"
 


# ── Document Upload ──────────────────────────────────
 
@router.post("/documents/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    allowed = {".pdf", ".md", ".html", ".htm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await ingest_document(filepath, session_id=session_id)

    return {
        "document_id": result["document_id"],
        "filename": result["filename"],
        "chunks_created": result["chunks_created"],
    }
 
 