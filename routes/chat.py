import json
import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from llm.agents import run_pipeline, stream_pipeline, summarize_conversation, extract_memories
from llm.schema import ChatRequest, ChatResponse, SourceInfo
from sqlalchemy.ext.asyncio import AsyncSession
from services.session_management import (
    get_session, add_message,
    get_recent_messages, save_memory,
    update_session_summary,
    save_agent_trace, get_session_traces,
)
from db.sql_database import get_db
from config import get_settings

router = APIRouter()


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
    _t0 = time.perf_counter()
    result = await run_pipeline(
        session_id=req.session_id,
        user_id=session.user_id,
        query=req.message,
        conversation_history=history,
        conversation_summary=session.summary or "",
    )
    response_time_ms = (time.perf_counter() - _t0) * 1000

    answer = result.get("answer", "I couldn't generate a response.")

    # Save assistant message
    await add_message(db, req.session_id, "assistant", answer)

    # Refresh session to get updated turn count
    session = await get_session(db, req.session_id)

    # Summarize if turn threshold reached
    if session.turn_count > 0 and session.turn_count % settings.SUMMARY_TRIGGER_TURNS == 0:
        all_msgs = await get_recent_messages(db, req.session_id, limit=30)
        msg_dicts = [{"role": m.role, "content": m.content} for m in all_msgs]
        summary = await summarize_conversation(msg_dicts, session.summary or "")
        await update_session_summary(db, req.session_id, summary)

    # Extract memories (runs on every exchange, lightweight)
    memories = await extract_memories(req.message, answer)
    for mem in memories:
        await save_memory(db, session.user_id, mem["memory_type"], mem["content"])

    await save_agent_trace(
        db,
        session_id=req.session_id,
        query_intent=str(result.get("query_intent", "")),
        retrieval_strategy=str(result.get("retrieval_strategy", "")),
        rewritten_query=result.get("rewritten_query"),
        sub_questions=result.get("sub_questions"),
        nodes_visited=None,
        response_time_ms=response_time_ms,
    )

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


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Streaming variant of /chat. Returns Server-Sent Events:
      - data: <token>          — one LLM token per event
      - data: [DONE] <json>   — final event with sources JSON
    """
    session = await get_session(db, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    settings = get_settings()
    await add_message(db, req.session_id, "user", req.message)

    recent = await get_recent_messages(db, req.session_id, limit=settings.MEMORY_WINDOW_SIZE * 2)
    history = [{"role": m.role, "content": m.content} for m in recent]

    async def event_generator():
        final_answer = ""
        final_sources = []
        done_meta = {}
        _t0 = time.perf_counter()

        async for chunk in stream_pipeline(
            session_id=req.session_id,
            user_id=session.user_id,
            query=req.message,
            conversation_history=history,
            conversation_summary=session.summary or "",
        ):
            if isinstance(chunk, dict) and chunk.get("done"):
                done_meta = chunk
                final_answer = chunk["answer"]
                final_sources = [
                    {"document_name": s.document_name, "snippet": s.snippet, "score": s.score}
                    for s in chunk.get("sources", [])
                ]
                yield f"data: [DONE] {json.dumps({'sources': final_sources})}\n\n"
            else:
                yield f"data: {chunk}\n\n"

        response_time_ms = (time.perf_counter() - _t0) * 1000

        # Persist after stream completes
        await add_message(db, req.session_id, "assistant", final_answer)

        session_updated = await get_session(db, req.session_id)
        if session_updated and session_updated.turn_count % settings.SUMMARY_TRIGGER_TURNS == 0:
            all_msgs = await get_recent_messages(db, req.session_id, limit=30)
            msg_dicts = [{"role": m.role, "content": m.content} for m in all_msgs]
            summary = await summarize_conversation(msg_dicts, session_updated.summary or "")
            await update_session_summary(db, req.session_id, summary)

        memories = await extract_memories(req.message, final_answer)
        for mem in memories:
            await save_memory(db, session.user_id, mem["memory_type"], mem["content"])

        await save_agent_trace(
            db,
            session_id=req.session_id,
            query_intent=str(done_meta.get("query_intent", "")),
            retrieval_strategy=str(done_meta.get("retrieval_strategy", "")),
            rewritten_query=done_meta.get("rewritten_query"),
            sub_questions=done_meta.get("sub_questions"),
            nodes_visited=None,
            response_time_ms=response_time_ms,
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/chat/trace/{session_id}")
async def get_trace(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the agent decision log for a session — intent, retrieval strategy,
    rewritten query, and sub-questions for each turn.
    """
    session = await get_session(db, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    traces = await get_session_traces(db, session_id)
    return [
        {
            "turn": i + 1,
            "query_intent": t.query_intent,
            "retrieval_strategy": t.retrieval_strategy,
            "rewritten_query": t.rewritten_query,
            "sub_questions": json.loads(t.sub_questions or "[]"),
            "nodes_visited": json.loads(t.nodes_visited or "[]"),
            "response_time_ms": t.response_time_ms,
            "created_at": t.created_at,
        }
        for i, t in enumerate(reversed(traces))
    ]
