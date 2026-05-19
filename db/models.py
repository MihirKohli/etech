from datetime import datetime
from uuid6 import uuid7
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: uuid7().hex)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default= datetime.now())
    summary = Column(Text, default="")
    turn_count = Column(Integer, default=0)
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: uuid7().hex)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)       # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default= datetime.now())
    session = relationship("Session", back_populates="messages")


class ConversationMemory(Base):
    """Cross-session long-term memory per user."""
    __tablename__ = "conversation_memory"

    id = Column(String, primary_key=True, default=lambda: uuid7().hex)
    user_id = Column(String, nullable=False, index=True)
    memory_type = Column(String, nullable=False)   # "preference" | "fact" | "summary"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default= datetime.now())

