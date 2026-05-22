import uuid
from datetime import datetime, date
from sqlalchemy import String, Text, Boolean, Integer, Date, DateTime, ForeignKey, Numeric, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class CropType(Base):
    __tablename__ = "crop_types"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    name_fa: Mapped[str] = mapped_column(String(150), nullable=False)
    category: Mapped[str] = mapped_column(String(50))  # vegetable/fruit/grain/herb
    cycle_days: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)


class LandBlock(Base):
    __tablename__ = "land_blocks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    area_ha: Mapped[float] = mapped_column(Numeric(8, 3), nullable=False)
    soil_type: Mapped[str | None] = mapped_column(String(50))
    irrigation_type: Mapped[str | None] = mapped_column(String(50))
    gps_polygon: Mapped[dict | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    farm: Mapped["Farm"] = relationship(back_populates="blocks")  # type: ignore
    cycles: Mapped[list["CropCycle"]] = relationship(back_populates="block")


class CropCycle(Base):
    __tablename__ = "crop_cycles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    block_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("land_blocks.id", ondelete="CASCADE"))
    crop_type_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crop_types.id"))
    season: Mapped[str] = mapped_column(String(20))  # spring/summer/fall/winter
    year: Mapped[int] = mapped_column(Integer)
    plant_date: Mapped[date | None] = mapped_column(Date)
    expected_harvest_date: Mapped[date | None] = mapped_column(Date)
    actual_harvest_date: Mapped[date | None] = mapped_column(Date)
    target_yield_kg: Mapped[float | None] = mapped_column(Numeric(12, 2))
    actual_yield_kg: Mapped[float | None] = mapped_column(Numeric(12, 2))
    status: Mapped[str] = mapped_column(String(30), default="planned")  # planned/planting/growing/harvested/closed
    total_cost_tomans: Mapped[float | None] = mapped_column(Numeric(15, 2))
    cost_per_kg: Mapped[float | None] = mapped_column(Numeric(12, 2))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    block: Mapped["LandBlock"] = relationship(back_populates="cycles")
    crop_type: Mapped["CropType"] = relationship()
    inputs: Mapped[list["CropInput"]] = relationship(back_populates="cycle", cascade="all, delete-orphan")


class CropInput(Base):
    __tablename__ = "crop_inputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cycle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("crop_cycles.id", ondelete="CASCADE"))
    input_category: Mapped[str] = mapped_column(String(50))  # seed/fertilizer/pesticide/water/labor/machinery/overhead/transport/waste
    input_type: Mapped[str] = mapped_column(String(200))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(12, 3))
    unit: Mapped[str] = mapped_column(String(30))
    unit_price: Mapped[float] = mapped_column(Numeric(15, 2))
    total_price: Mapped[float] = mapped_column(Numeric(15, 2))
    supplier: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cycle: Mapped["CropCycle"] = relationship(back_populates="inputs")
