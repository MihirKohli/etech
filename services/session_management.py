from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import Session, Message, ConversationMemory


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
    mem = ConversationMemory(user_id=user_id, memory_type=memory_type, content=content)
    db.add(mem)
    await db.commit()
    return mem


async def get_user_memories(db: AsyncSession, user_id: str, limit: int = 20) -> list[ConversationMemory]:
    result = await db.execute(
        select(ConversationMemory)
        .where(ConversationMemory.user_id == user_id)
        .order_by(ConversationMemory.importance.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_session_summary(db: AsyncSession, session_id: str, summary: str):
    session = await db.get(Session, session_id)
    if session:
        session.summary = summary
        await db.commit()