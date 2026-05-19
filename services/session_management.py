import json
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import Session, Message, ConversationMemory, AgentTrace
from config import get_settings

_settings = get_settings()


async def create_session(db: AsyncSession, user_id: str) -> Session:
    session = Session(user_id=user_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    return await db.get(Session, session_id)


async def list_sessions(db: AsyncSession, user_id: str) -> list[dict]:
    sessions_result = await db.execute(
        select(Session).where(Session.user_id == user_id).order_by(Session.created_at.desc())
    )
    sessions = list(sessions_result.scalars().all())

    rows = []
    for s in sessions:
        first_msg_result = await db.execute(
            select(Message)
            .where(Message.session_id == s.id, Message.role == "user")
            .order_by(Message.created_at)
            .limit(1)
        )
        first_msg = first_msg_result.scalar_one_or_none()
        preview = " ".join(first_msg.content.split()[:5]) if first_msg else "New conversation"
        rows.append({"session": s, "preview": preview})
    return rows


async def add_message(db: AsyncSession, session_id: str, role: str, content: str) -> Message:
    msg = Message(session_id=session_id, role=role, content=content)
    db.add(msg)

    # bump turn count
    session = await db.get(Session, session_id)
    if session:
        session.turn_count += 1

    await db.commit()
    await db.refresh(msg)
    return msg


async def get_recent_messages(db: AsyncSession, session_id: str, limit: int = 10) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()
    return messages


async def save_memory(db: AsyncSession, user_id: str, memory_type: str, content: str):
    # Skip exact duplicates
    existing = await db.execute(
        select(ConversationMemory).where(
            ConversationMemory.user_id == user_id,
            ConversationMemory.memory_type == memory_type,
            ConversationMemory.content == content,
        ).limit(1)
    )
    if existing.scalar_one_or_none():
        return None

    mem = ConversationMemory(user_id=user_id, memory_type=memory_type, content=content)
    db.add(mem)
    await db.commit()

    # Prune oldest entries beyond MEMORY_MAX_ENTRIES per user
    count_result = await db.execute(
        select(func.count()).where(ConversationMemory.user_id == user_id)
    )
    total = count_result.scalar()
    if total > _settings.MEMORY_MAX_ENTRIES:
        overflow = total - _settings.MEMORY_MAX_ENTRIES
        oldest_ids_result = await db.execute(
            select(ConversationMemory.id)
            .where(ConversationMemory.user_id == user_id)
            .order_by(ConversationMemory.created_at.asc())
            .limit(overflow)
        )
        oldest_ids = [row[0] for row in oldest_ids_result]
        await db.execute(
            delete(ConversationMemory).where(ConversationMemory.id.in_(oldest_ids))
        )
        await db.commit()

    return mem


async def get_user_memories(db: AsyncSession, user_id: str, limit: int = 20) -> list[ConversationMemory]:
    result = await db.execute(
        select(ConversationMemory)
        .where(ConversationMemory.user_id == user_id)
        .order_by(ConversationMemory.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_session_summary(db: AsyncSession, session_id: str, summary: str):
    session = await db.get(Session, session_id)
    if session:
        session.summary = summary
        await db.commit()


async def save_agent_trace(
    db: AsyncSession,
    session_id: str,
    query_intent: str | None,
    retrieval_strategy: str | None,
    rewritten_query: str | None,
    sub_questions: list[str] | None,
    nodes_visited: list[str] | None,
    response_time_ms: float | None = None,
):
    trace = AgentTrace(
        session_id=session_id,
        query_intent=query_intent,
        retrieval_strategy=retrieval_strategy,
        rewritten_query=rewritten_query,
        sub_questions=json.dumps(sub_questions or []),
        nodes_visited=json.dumps(nodes_visited or []),
        response_time_ms=response_time_ms,
    )
    db.add(trace)
    await db.commit()
    return trace


async def get_session_traces(db: AsyncSession, session_id: str, limit: int = 50) -> list[AgentTrace]:
    result = await db.execute(
        select(AgentTrace)
        .where(AgentTrace.session_id == session_id)
        .order_by(AgentTrace.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())