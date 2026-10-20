"""SQLAlchemy persistence for BillShield's minimized bill and message audit data."""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


DATABASE_PATH = Path(__file__).resolve().parent / ".billshield_sqlalchemy.sqlite3"
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Bill(Base):
    __tablename__ = "bills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    provider_name: Mapped[str] = mapped_column(String(200), nullable=False)
    billing_period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    billing_period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    # Intentionally remains NULL: BillShield never persists uploaded-file paths.
    file_reference: Mapped[str | None] = mapped_column(String(500), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    line_items: Mapped[list[LineItem]] = relationship(
        back_populates="bill", cascade="all, delete-orphan", passive_deletes=True
    )
    promises: Mapped[list[Promise]] = relationship(
        back_populates="bill", cascade="all, delete-orphan", passive_deletes=True
    )


class LineItem(Base):
    __tablename__ = "line_items"
    __table_args__ = (Index("ix_line_items_charge_fingerprint", "charge_fingerprint"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    bill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("bills.id", ondelete="CASCADE"), nullable=False, index=True
    )
    item_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    billed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    charge_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    verification_status: Mapped[str] = mapped_column(String(40), nullable=False, default="UNREVIEWED")
    # Only a fixed, code-generated label is saved here; never raw OCR or user text.
    verification_evidence: Mapped[str | None] = mapped_column(String(200), nullable=True)

    bill: Mapped[Bill] = relationship(back_populates="line_items")
    messages: Mapped[list[Message]] = relationship(
        back_populates="line_item", cascade="all, delete-orphan", passive_deletes=True
    )


class Promise(Base):
    __tablename__ = "promises"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    bill_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bills.id", ondelete="CASCADE"), nullable=True, index=True
    )
    # Random UUID generated per app session; contains no identity or account data.
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    provider_name: Mapped[str] = mapped_column(String(200), nullable=False)
    service_plan_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    promised_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_until: Mapped[date | None] = mapped_column(Date, nullable=True)
    terms_summary: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    bill: Mapped[Bill | None] = relationship(back_populates="promises")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT','COPIED','SENT_USER_REPORTED','FAILED','RESOLVED')",
            name="ck_messages_status",
        ),
        CheckConstraint("word_edit_distance IS NULL OR word_edit_distance >= 0", name="ck_messages_edit_distance"),
        CheckConstraint(
            "((status IN ('SENT_USER_REPORTED','RESOLVED') AND sent_at IS NOT NULL "
            "AND sent_source_type = 'user_reported') OR "
            "(status IN ('DRAFT','COPIED','FAILED') AND sent_at IS NULL AND sent_source_type IS NULL))",
            name="ck_messages_sent_requires_user_report",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    # This FK is the critical message -> disputed line item link from the ERD.
    line_item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("line_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="DRAFT")
    word_edit_distance: Mapped[int | None] = mapped_column(nullable=True)
    dispatch_channel: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    sent_source_type: Mapped[str | None] = mapped_column(String(40), nullable=True, default=None)

    line_item: Mapped[LineItem] = relationship(back_populates="messages")


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db(database_url: str | None = None) -> sessionmaker[Session]:
    """Create the local SQLite schema and return the configured session maker."""
    global _engine, _session_factory
    if database_url is None:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{DATABASE_PATH.as_posix()}"
    if _engine is not None:
        _engine.dispose()
    connect_args = {"check_same_thread": False, "timeout": 15} if database_url.startswith("sqlite") else {}
    _engine = create_engine(database_url, connect_args=connect_args, future=True)
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, class_=Session)
    Base.metadata.create_all(_engine)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    if _session_factory is None:
        init_db()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _safe_label(value: str | None, limit: int = 200) -> str:
    """Remove common PII patterns before storing labels supplied by OCR/AI."""
    value = (value or "Unknown").strip()
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[redacted]", value)
    value = re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)", "[redacted]", value)
    value = re.sub(r"\b(?:account|acct|consumer|customer|card|phone|mobile|meter|service)\s*(?:no\.?|number|#|:)?\s*[A-Z0-9-]{5,}\b", "[redacted]", value, flags=re.I)
    value = re.sub(r"\b\d{12,19}\b", "[redacted]", value)
    return value[:limit] or "Unknown"


def charge_fingerprint(provider_name: str, item_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{provider_name} {item_name}".casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def persist_verified_bill(
    *, session_id: str, provider_name: str, billing_period_start: date | None,
    billing_period_end: date | None, actual_total: Decimal, expected_total: Decimal,
    items: list[dict], promise_window: dict,
) -> tuple[str, dict, dict[str, str], dict[str, str]]:
    """Persist verified bill facts, its promise, and line items; never persist raw content."""
    safe_provider = _safe_label(provider_name)
    safe_session_id = str(session_id)
    history: dict = {}
    line_ids: dict[str, str] = {}
    price_hikes: dict[str, str] = {}
    with session_scope() as session:
        bill = Bill(
            provider_name=safe_provider,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            total_amount=actual_total,
            currency="INR",
            file_reference=None,
        )
        session.add(bill)
        session.flush()
        for item in items:
            safe_item_name = _safe_label(item["name"])
            fingerprint = charge_fingerprint(safe_provider, safe_item_name)
            old = session.scalar(
                select(LineItem)
                .join(Bill, Bill.id == LineItem.bill_id)
                .where(LineItem.charge_fingerprint == fingerprint, LineItem.bill_id != bill.id)
                .order_by(Bill.created_at.desc())
                .limit(1)
            )
            if old is not None:
                history[item["name"]] = {"amount": str(old.billed_amount), "name": old.item_name, "date": str(old.bill.created_at)}
                if Decimal(item["amount"]) > Decimal(old.billed_amount):
                    price_hikes[item["name"]] = str(old.billed_amount)
            line = LineItem(
                bill_id=bill.id,
                item_name=safe_item_name,
                description=None,
                billed_amount=item["amount"],
                charge_fingerprint=fingerprint,
                verification_status="INCLUDED_IN_TOTAL",
                verification_evidence="Included in code-calculated bill total.",
            )
            session.add(line)
            session.flush()
            line_ids[item["name"]] = line.id

        session.add(Promise(
            bill_id=bill.id,
            session_id=safe_session_id,
            provider_name=safe_provider,
            service_plan_name=None,
            promised_amount=expected_total,
            valid_from=date.fromisoformat(promise_window["valid_from"]) if promise_window.get("valid_from") else None,
            valid_until=date.fromisoformat(promise_window["valid_until"]) if promise_window.get("valid_until") else None,
            terms_summary="Expected bill total supplied by the user.",
        ))
        session.flush()
        bill_id = bill.id
    return bill_id, history, line_ids, price_hikes


def create_message_record(line_item_id: str) -> str:
    with session_scope() as session:
        message = Message(line_item_id=line_item_id, status="DRAFT", dispatch_channel="official_web_form")
        session.add(message)
        session.flush()
        return message.id


def saved_records_snapshot(limit: int = 100) -> dict[str, list[list[str]] | dict[str, int]]:
    """Return display-safe rows for the local Saved Records panel; never return message text."""
    def stamp(value: datetime | None) -> str:
        return value.isoformat(timespec="seconds") if value else ""

    with session_scope() as session:
        counts = {
            "Bills": session.scalar(select(func.count()).select_from(Bill)) or 0,
            "Line items": session.scalar(select(func.count()).select_from(LineItem)) or 0,
            "Promises": session.scalar(select(func.count()).select_from(Promise)) or 0,
            "Messages": session.scalar(select(func.count()).select_from(Message)) or 0,
        }
        bills = session.execute(
            select(Bill).order_by(Bill.created_at.desc()).limit(limit)
        ).scalars().all()
        line_items = session.execute(
            select(Bill.provider_name, LineItem.item_name, LineItem.billed_amount,
                   LineItem.verification_status, LineItem.bill_id)
            .join(Bill, Bill.id == LineItem.bill_id)
            .order_by(Bill.created_at.desc()).limit(limit)
        ).all()
        promises = session.execute(
            select(Promise.provider_name, Promise.service_plan_name, Promise.promised_amount,
                   Promise.valid_from, Promise.valid_until, Promise.bill_id, Promise.created_at)
            .order_by(Promise.created_at.desc()).limit(limit)
        ).all()
        messages = session.execute(
            select(LineItem.item_name, Message.status, Message.word_edit_distance,
                   Message.dispatch_channel, Message.created_at, Message.sent_at,
                   Message.sent_source_type)
            .join(LineItem, LineItem.id == Message.line_item_id)
            .order_by(Message.created_at.desc()).limit(limit)
        ).all()

    return {
        "counts": counts,
        "bills": [[b.provider_name, f"{b.billing_period_start or ''} – {b.billing_period_end or ''}".strip(" –"), f"₹{b.total_amount:,.2f} {b.currency}", stamp(b.created_at)] for b in bills],
        "line_items": [[provider, name, f"₹{amount:,.2f}", status, bill_id] for provider, name, amount, status, bill_id in line_items],
        "promises": [[provider, plan or "", f"₹{amount:,.2f}", str(valid_from or ""), str(valid_until or ""), bill_id or "Not linked", stamp(created)] for provider, plan, amount, valid_from, valid_until, bill_id, created in promises],
        "messages": [[item_name, status, str(edit_distance if edit_distance is not None else ""), channel or "", stamp(created), stamp(sent_at), source or ""] for item_name, status, edit_distance, channel, created, sent_at, source in messages],
    }


_TRANSITIONS = {
    "DRAFT": {"COPIED", "FAILED"},
    "COPIED": {"SENT_USER_REPORTED", "FAILED"},
    "SENT_USER_REPORTED": {"RESOLVED"},
    "FAILED": set(),
    "RESOLVED": set(),
}


def update_message_status(
    message_id: str, old_status: str, new_status: str, *,
    user_confirmed_sent: bool = False, word_edit_distance: int | None = None,
) -> bool:
    """Apply a valid audit transition; sent_at is written only on explicit confirmation."""
    same_state_edit = old_status == new_status == "DRAFT" and word_edit_distance is not None
    if not same_state_edit and new_status not in _TRANSITIONS.get(old_status, set()):
        raise ValueError(f"Invalid message status transition: {old_status} -> {new_status}.")
    if new_status == "SENT_USER_REPORTED" and not user_confirmed_sent:
        raise ValueError("Sending must be explicitly confirmed by the user.")
    values: dict = {"status": new_status}
    if word_edit_distance is not None:
        if word_edit_distance < 0:
            raise ValueError("word_edit_distance cannot be negative.")
        values["word_edit_distance"] = word_edit_distance
    if new_status == "SENT_USER_REPORTED":
        values["sent_at"] = utc_now()
        values["sent_source_type"] = "user_reported"
    with session_scope() as session:
        result = session.execute(
            update(Message)
            .where(Message.id == message_id, Message.status == old_status)
            .values(**values)
        )
        return result.rowcount == 1
