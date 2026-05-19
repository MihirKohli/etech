from fastapi import APIRouter, Depends
from db.sql_database import get_db
from services.session_management import get_user_memories
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
 
UPLOAD_DIR = "./data/uploads"
 
# Memories
 
@router.get("/memories/{user_id}")
async def get_memories(user_id: str, db: AsyncSession = Depends(get_db)):
    memories = await get_user_memories(db, user_id)
    return [
        {
            "memory_type": m.memory_type,
            "content": m.content,
            "created_at": m.created_at,
        }
        for m in memories
    ]
