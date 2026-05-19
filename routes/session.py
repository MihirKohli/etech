from fastapi import Depends, HTTPException, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from db.sql_database import get_db
from services.session_management import create_session, get_session, list_sessions


router = APIRouter()

@router.post("/sessions")
async def create_new_session(user_id: str, db: AsyncSession = Depends(get_db)):
    session = await create_session(db, user_id)
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "created_at": session.created_at,
    }
 


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await get_session(db, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.id,
        "user_id": session.user_id,
        "turn_count": session.turn_count,
        "summary": session.summary,
        "created_at": session.created_at,
    }

@router.get("/sessions")
async def list_user_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    sessions = await list_sessions(db, user_id)
    return [
        {"session_id": s.id, "turn_count": s.turn_count, "created_at": s.created_at}
        for s in sessions
    ]
 
