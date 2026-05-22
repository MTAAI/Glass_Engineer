from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.core.cost_engine import crop_cycle_cost
from app.models.machinery import Machinery, MachineryUsage


router = APIRouter(prefix="/api/v1/machinery", tags=["machinery"])


class MachineryOut(BaseModel):
    id: UUID
    farm_id: UUID
    name: str
    type: str
    purchase_price: float | None = None
    useful_hours: int | None = None
    fuel_l_per_hour: float | None = None
    operator_hourly: float | None = None
    hourly_rate: float | None = None
    total_hours_used: int
    active: bool
    notes: str | None = None

    model_config = {"from_attributes": True}


class MachineryCreate(BaseModel):
    farm_id: UUID
    name: str
    type: str
    purchase_price: float | None = None
    useful_hours: int | None = None
    fuel_l_per_hour: float | None = None
    operator_hourly: float | None = None
    notes: str | None = None


class UsageOut(BaseModel):
    id: UUID
    machinery_id: UUID
    cycle_id: UUID | None = None
    operation: str
    date: date
    hours: float
    hourly_rate: float
    total_cost: float
    fuel_used_l: float | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class UsageCreate(BaseModel):
    cycle_id: UUID | None = None
    operation: str
    date: date
    hours: float
    fuel_used_l: float | None = None
    notes: str | None = None


def _compute_hourly_rate(m: Machinery, diesel_price: float = 25_000) -> float:
    depreciation = (float(m.purchase_price) / m.useful_hours) if (m.purchase_price and m.useful_hours) else 0
    fuel = float(m.fuel_l_per_hour) * diesel_price if m.fuel_l_per_hour else 0
    operator = float(m.operator_hourly) if m.operator_hourly else 0
    return depreciation + fuel + operator


@router.get("", response_model=list[MachineryOut])
def list_machinery(farm_id: UUID | None = None, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[MachineryOut]:
    stmt = select(Machinery).where(Machinery.active.is_(True))
    if farm_id:
        stmt = stmt.where(Machinery.farm_id == farm_id)
    return [MachineryOut.model_validate(m) for m in db.scalars(stmt).all()]


@router.post("", response_model=MachineryOut, status_code=201)
def create_machinery(body: MachineryCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> MachineryOut:
    m = Machinery(**body.model_dump())
    m.hourly_rate = _compute_hourly_rate(m)
    db.add(m)
    db.commit()
    db.refresh(m)
    return MachineryOut.model_validate(m)


@router.get("/{machinery_id}/usage", response_model=list[UsageOut])
def list_usage(machinery_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[UsageOut]:
    rows = db.scalars(
        select(MachineryUsage).where(MachineryUsage.machinery_id == machinery_id).order_by(MachineryUsage.date.desc())
    ).all()
    return [UsageOut.model_validate(r) for r in rows]


@router.post("/{machinery_id}/usage", response_model=UsageOut, status_code=201)
def add_usage(machinery_id: UUID, body: UsageCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> UsageOut:
    m = db.get(Machinery, machinery_id)
    if not m:
        raise HTTPException(status_code=404, detail="ماشین‌آلات یافت نشد")

    rate = float(m.hourly_rate or _compute_hourly_rate(m))
    total = body.hours * rate

    usage = MachineryUsage(
        machinery_id=machinery_id,
        hourly_rate=rate,
        total_cost=total,
        **body.model_dump(),
    )
    m.total_hours_used = (m.total_hours_used or 0) + int(body.hours)
    db.add_all([usage, m])
    db.commit()
    db.refresh(usage)

    if body.cycle_id:
        crop_cycle_cost(db, body.cycle_id)
        db.commit()

    return UsageOut.model_validate(usage)
