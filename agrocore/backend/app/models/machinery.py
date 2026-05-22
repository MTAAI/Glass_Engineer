import uuid
from datetime import datetime, date
from sqlalchemy import String, Text, Boolean, Integer, Date, DateTime, ForeignKey, Numeric, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Machinery(Base):
    __tablename__ = "machinery"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[str] = mapped_column(String(100))  # tractor/harvester/disc/plough/sprayer
    purchase_price: Mapped[float | None] = mapped_column(Numeric(15, 2))
    useful_hours: Mapped[int | None] = mapped_column(Integer)
    fuel_l_per_hour: Mapped[float | None] = mapped_column(Numeric(6, 2))
    operator_hourly: Mapped[float | None] = mapped_column(Numeric(12, 2))
    hourly_rate: Mapped[float | None] = mapped_column(Numeric(12, 2))
    total_hours_used: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text)


class MachineryUsage(Base):
    __tablename__ = "machinery_usage"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    machinery_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("machinery.id", ondelete="CASCADE"))
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("crop_cycles.id"))
    operation: Mapped[str] = mapped_column(String(200))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    hours: Mapped[float] = mapped_column(Numeric(6, 2))
    hourly_rate: Mapped[float] = mapped_column(Numeric(12, 2))
    total_cost: Mapped[float] = mapped_column(Numeric(15, 2))
    fuel_used_l: Mapped[float | None] = mapped_column(Numeric(8, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
