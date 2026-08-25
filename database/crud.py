"""
CRUD operations and prompt context builder for Khedut Voice AI.
Farmer profile is populated automatically from conversation — never pre-set with fake data.
"""

import uuid
from typing import List, Optional
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from .models import FarmerProfile, Conversation, Message, utc_now


# ─── Farmer Profile Operations ────────────────────────────────────────────────

async def get_or_create_default_profile(db: AsyncSession) -> FarmerProfile:
    """
    Retrieve the primary farmer profile, or create a completely blank one.
    Profile fields are filled automatically by extract_and_update_profile_from_conversation()
    as the farmer speaks — never pre-populated with assumed data.
    """
    stmt = select(FarmerProfile).order_by(FarmerProfile.id).limit(1)
    result = await db.execute(stmt)
    profile = result.scalar_one_or_none()

    if not profile:
        # Blank profile — nothing assumed
        profile = FarmerProfile()
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return profile


async def update_farmer_profile(db: AsyncSession, profile_id: int, data: dict) -> Optional[FarmerProfile]:
    """Update profile fields (only non-None values are applied)."""
    stmt = select(FarmerProfile).where(FarmerProfile.id == profile_id)
    result = await db.execute(stmt)
    profile = result.scalar_one_or_none()
    if not profile:
        return None

    for key, value in data.items():
        if hasattr(profile, key) and value is not None:
            # For lists (crops), merge instead of replace if field already has data
            if key == "crops" and isinstance(value, list):
                existing = profile.crops or []
                merged = list(dict.fromkeys(existing + value))  # deduplicated union
                setattr(profile, key, merged)
            else:
                setattr(profile, key, value)

    profile.updated_at = utc_now()
    await db.commit()
    await db.refresh(profile)
    return profile


# ─── Conversation Operations ──────────────────────────────────────────────────

async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Conversation:
    """Find existing conversation by ID or create a new session with an isolated blank profile."""
    if conversation_id:
        stmt = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(
                selectinload(Conversation.messages),
                selectinload(Conversation.farmer),
            )
        )
        result = await db.execute(stmt)
        conv = result.scalar_one_or_none()
        if conv:
            return conv

    # Create new isolated blank profile specifically for this new conversation session
    new_profile = FarmerProfile()
    db.add(new_profile)
    await db.flush()

    new_id = conversation_id or str(uuid.uuid4())
    conv = Conversation(
        id=new_id,
        farmer_id=new_profile.id,
        title=title or "નવી ખેતી વાતચીત",
        is_active=True,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def list_conversations(db: AsyncSession, limit: int = 20) -> List[Conversation]:
    """List recent conversations ordered by last activity."""
    stmt = (
        select(Conversation)
        .execution_options(populate_existing=True)
        .options(
            selectinload(Conversation.messages),
            selectinload(Conversation.farmer),
        )
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_conversation(db: AsyncSession, conversation_id: str) -> Optional[Conversation]:
    """Get single conversation with all its messages and linked farmer profile."""
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .execution_options(populate_existing=True)
        .options(
            selectinload(Conversation.messages),
            selectinload(Conversation.farmer),
        )
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def delete_conversation(db: AsyncSession, conversation_id: str) -> bool:
    """Delete a conversation and its isolated profile."""
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.farmer))
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        return False
    if conv.farmer:
        await db.delete(conv.farmer)
    await db.delete(conv)
    await db.commit()
    return True


# ─── Message Operations ───────────────────────────────────────────────────────

async def add_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    audio_format: Optional[str] = None,
) -> Message:
    """Add a message turn and update the conversation's timestamp and auto-title."""
    conv = await get_or_create_conversation(db, conversation_id=conversation_id)

    msg = Message(
        conversation_id=conv.id,
        role=role,
        content=content.strip(),
        audio_format=audio_format,
        timestamp=utc_now(),
    )
    db.add(msg)

    # Auto-set conversation title from first user message
    if role == "user" and conv.title in ("નવી ખેતી વાતચીત", ""):
        clean_text = content.strip().replace("\n", " ")
        if clean_text:
            conv.title = clean_text[:40] + ("..." if len(clean_text) > 40 else "")

    conv.updated_at = utc_now()
    await db.commit()
    await db.refresh(msg)
    return msg


# ─── Context Builder for Gemini Live ──────────────────────────────────────────

async def build_conversation_context(
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    max_messages: int = 10,
) -> str:
    """
    Builds a concise context string to inject into Gemini's system prompt.
    Only includes profile fields that are actually known from the current conversation.
    Includes recent message history for continuity.
    Returns an empty string if nothing is known yet.
    """
    conv = None
    profile = None
    if conversation_id:
        conv = await get_conversation(db, conversation_id)
        if conv and conv.farmer:
            profile = conv.farmer

    # Collect only the known profile fields
    known_facts = []
    if profile:
        if profile.name:
            known_facts.append(f"ખેડૂતનું નામ: {profile.name}")
        if profile.village:
            known_facts.append(f"ગામ: {profile.village}")
        if profile.district:
            known_facts.append(f"જિલ્લો: {profile.district}")
        if profile.land_acres:
            known_facts.append(f"જમીન: {profile.land_acres} એકર")
        if profile.crops:
            crops_str = ", ".join(profile.crops) if isinstance(profile.crops, list) else str(profile.crops)
            known_facts.append(f"ખેડૂત પાક: {crops_str}")
        if profile.soil_type:
            known_facts.append(f"જમીનનો પ્રકાર: {profile.soil_type}")
        if profile.farming_type:
            known_facts.append(f"ખેતી પ્રકાર: {profile.farming_type}")
        if profile.notes:
            known_facts.append(f"નોંધ: {profile.notes}")

    # Collect recent message history
    history_lines = []
    recent_user_queries = []
    if conv and conv.messages:
        recent = conv.messages[-max_messages:]
        for m in recent:
            sender = "ખેડૂત" if m.role == "user" else "AI"
            history_lines.append(f"{sender}: {m.content}")
            if m.role == "user" and m.content.strip():
                recent_user_queries.append(m.content.strip())

    # ── RAG Knowledge Base Retrieval ──────────────────────────────────────────
    rag_context = ""
    try:
        from rag.retriever import build_rag_context
        query_text = recent_user_queries[-1] if recent_user_queries else ""
        if query_text:
            rag_context = await build_rag_context(
                query=query_text,
                crops=profile.crops if (profile and isinstance(profile.crops, list)) else None,
                max_chunks=2,
            )
    except Exception:
        pass

    # Build context string — only if something is genuinely known
    parts = []
    if known_facts:
        parts.append("### આ ખેડૂત વિશે વાતચીતમાં જાણવા મળેલ વિગતો:\n" + "\n".join(f"- {f}" for f in known_facts))
    if rag_context.strip():
        parts.append(rag_context.strip())
    if history_lines:
        parts.append("### અગાઉની વાતચીત:\n" + "\n".join(history_lines))

    return "\n\n".join(parts)
