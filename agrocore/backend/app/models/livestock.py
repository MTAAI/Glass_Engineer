import uuid
from datetime import datetime, date
from sqlalchemy import String, Text, Integer, Date, DateTime, ForeignKey, Numeric, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class LivestockGroup(Base):
    __tablename__ = "livestock_groups"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"))
    species: Mapped[str] = mapped_column(String(50))  # cow/sheep/goat/chicken/duck/goose/turkey/trout/carp/tilapia
    purpose: Mapped[str] = mapped_column(String(50))  # dairy/meat/egg/dual/wool
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    current_count: Mapped[int] = mapped_column(Integer, default=0)
    target_count: Mapped[int] = mapped_column(Integer, default=0)
    entry_date: Mapped[date | None] = mapped_column(Date)
    total_cost_to_date: Mapped[float] = mapped_column(Numeric(15, 2), default=0)
    avg_daily_production: Mapped[float | None] = mapped_column(Numeric(10, 3))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inputs: Mapped[list["LivestockInput"]] = relationship(back_populates="group", cascade="all, delete-orphan")
    production: Mapped[list["LivestockDailyProduction"]] = relationship(back_populates="group", cascade="all, delete-orphan")
    events: Mapped[list["LivestockEvent"]] = relationship(back_populates="group", cascade="all, delete-orphan")


class LivestockInput(Base):
    __tablename__ = "livestock_inputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("livestock_groups.id", ondelete="CASCADE"))
    input_category: Mapped[str] = mapped_column(String(50))  # feed/medicine/vet/utilities/labor/bedding
    input_name: Mapped[str] = mapped_column(String(200))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(12, 3))
    unit: Mapped[str] = mapped_column(String(30))
    unit_price: Mapped[float] = mapped_column(Numeric(15, 2))
    total_price: Mapped[float] = mapped_column(Numeric(15, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    group: Mapped["LivestockGroup"] = relationship(back_populates="inputs")


class LivestockDailyProduction(Base):
    __tablename__ = "livestock_daily_production"
    __table_args__ = (UniqueConstraint("group_id", "date", "production_type", name="uq_livestock_prod_day"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("livestock_groups.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    production_type: Mapped[str] = mapped_column(String(30))  # milk/egg/weight_gain/wool
    amount: Mapped[float] = mapped_column(Numeric(10, 3))
    quality_grade: Mapped[str | None] = mapped_column(String(10))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    group: Mapped["LivestockGroup"] = relationship(back_populates="production")


class LivestockEvent(Base):
    __tablename__ = "livestock_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("livestock_groups.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(50))  # birth/death/purchase/sale/slaughter/vaccination/illness
    date: Mapped[date] = mapped_column(Date, nullable=False)
    count: Mapped[int | None] = mapped_column(Integer)
    weight_kg: Mapped[float | None] = mapped_column(Numeric(10, 2))
    cost: Mapped[float | None] = mapped_column(Numeric(15, 2))
    vet_name: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    group: Mapped["LivestockGroup"] = relationship(back_populates="events")
