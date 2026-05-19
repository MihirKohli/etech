from fastapi import APIRouter, Depends, HTTPException
from agents import run_pipeline,summarize_conversation, extract_memories
from schema import ChatRequest, ChatResponse, SourceInfo
from sqlalchemy.ext.asyncio import AsyncSession
from services.session_management import (
    get_session, add_message,
    get_recent_messages, save_memory,
    update_session_summary,
)
from db.sql_database import get_db
from config import get_settings


router = APIRouter()



# ── Chat ─────────────────────────────────────────────
 
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    settings = get_settings()
 
    # Validate session
    session = await get_session(db, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
 
    # Save user message
    await add_message(db, req.session_id, "user", req.message)
 
    # Load recent history
    recent = await get_recent_messages(db, req.session_id, limit=settings.MEMORY_WINDOW_SIZE * 2)
    history = [{"role": m.role, "content": m.content} for m in recent]
 
    # Run the LangGraph pipeline
    result = run_pipeline(
        session_id=req.session_id,
        user_id=session.user_id,
        query=req.message,
        conversation_history=history,
        conversation_summary=session.summary or "",
    )
 
    answer = result.get("answer", "I couldn't generate a response.")
 
    # Save assistant message
    await add_message(db, req.session_id, "assistant", answer)
 
    # Refresh session to get updated turn count
    session = await get_session(db, req.session_id)
 
    # Summarize if turn threshold reached
    if session.turn_count > 0 and session.turn_count % settings.SUMMARY_TRIGGER_TURNS == 0:
        all_msgs = await get_recent_messages(db, req.session_id, limit=30)
        msg_dicts = [{"role": m.role, "content": m.content} for m in all_msgs]
        summary = summarize_conversation(msg_dicts, session.summary or "")
        await update_session_summary(db, req.session_id, summary)
 
    # Extract memories (runs on every exchange, lightweight)
    memories = extract_memories(req.message, answer)
    for mem in memories:
        await save_memory(db, session.user_id, mem["memory_type"], mem["content"])
 
    # Build response
    sources = [
        SourceInfo(
            document_name=s.document_name,
            snippet=s.snippet,
            score=s.score,
        )
        for s in result.get("sources", [])
    ]
 
    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        sources=sources,
    )
 
 