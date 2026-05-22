import uuid
from datetime import datetime, date
from sqlalchemy import String, Text, Boolean, Integer, Date, DateTime, ForeignKey, Numeric, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ProcessingLine(Base):
    __tablename__ = "processing_lines"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    input_product: Mapped[str] = mapped_column(String(200))
    output_product: Mapped[str] = mapped_column(String(200))
    conversion_ratio: Mapped[float] = mapped_column(Numeric(8, 3))
    capacity_kg_day: Mapped[int | None] = mapped_column(Integer)
    electricity_kwh_per_kg: Mapped[float | None] = mapped_column(Numeric(8, 4))
    water_l_per_kg: Mapped[float | None] = mapped_column(Numeric(8, 4))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text)


class ProcessingBatch(Base):
    __tablename__ = "processing_batches"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    line_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("processing_lines.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    input_source: Mapped[str] = mapped_column(String(50))  # farm_batch/purchase/dairy
    input_batch_id: Mapped[str | None] = mapped_column(String(60))
    input_qty_kg: Mapped[float] = mapped_column(Numeric(12, 3))
    output_qty_kg: Mapped[float | None] = mapped_column(Numeric(12, 3))
    waste_kg: Mapped[float] = mapped_column(Numeric(10, 3), default=0)
    input_cost: Mapped[float] = mapped_column(Numeric(15, 2), default=0)
    labor_cost: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    energy_cost: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    packaging_cost: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    other_cost: Mapped[float] = mapped_column(Numeric(12, 2), default=0)
    total_cost: Mapped[float] = mapped_column(Numeric(15, 2), default=0)
    cost_per_kg: Mapped[float | None] = mapped_column(Numeric(12, 2))
    status: Mapped[str] = mapped_column(String(30), default="planned")  # planned/running/done
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
