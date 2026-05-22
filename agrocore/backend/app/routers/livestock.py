from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.core.cost_engine import livestock_group_cost
from app.models.livestock import LivestockGroup, LivestockInput, LivestockDailyProduction, LivestockEvent
from app.models.user import User


router = APIRouter(prefix="/api/v1/livestock", tags=["livestock"])


# Business rule: trout banned in summer months (Tir/Mordad/Shahrivar ≈ Jul/Aug/Sep)
TROUT_BAN_MONTHS = {7, 8, 9}
SHEEP_MAX = 200


class GroupOut(BaseModel):
    id: UUID
    farm_id: UUID
    species: str
    purpose: str
    name: str
    current_count: int
    target_count: int
    entry_date: date | None = None
    total_cost_to_date: float | None = None
    avg_daily_production: float | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class GroupCreate(BaseModel):
    farm_id: UUID
    species: str
    purpose: str
    name: str
    current_count: int = 0
    target_count: int = 0
    entry_date: date | None = None
    notes: str | None = None


class InputOut(BaseModel):
    id: UUID
    group_id: UUID
    input_category: str
    input_name: str
    date: date
    quantity: float
    unit: str
    unit_price: float
    total_price: float
    notes: str | None = None

    model_config = {"from_attributes": True}


class InputCreate(BaseModel):
    input_category: str
    input_name: str
    date: date
    quantity: float
    unit: str
    unit_price: float
    notes: str | None = None


class ProductionOut(BaseModel):
    id: UUID
    group_id: UUID
    date: date
    production_type: str
    amount: float
    quality_grade: str | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class ProductionCreate(BaseModel):
    date: date
    production_type: str
    amount: float
    quality_grade: str | None = None
    notes: str | None = None


def _enforce_rules(body: GroupCreate, db: Session) -> None:
    if body.species == "trout" and body.entry_date and body.entry_date.month in TROUT_BAN_MONTHS:
        raise HTTPException(status_code=400, detail="ورود قزل‌آلا در تیر/مرداد/شهریور ممنوع است")
    if body.species == "sheep":
        existing = db.scalar(
            select(LivestockGroup).where(LivestockGroup.species == "sheep")
        )
        if existing and (existing.current_count + body.target_count) > SHEEP_MAX:
            raise HTTPException(status_code=400, detail=f"حداکثر تعداد گوسفند {SHEEP_MAX} رأس است")


@router.get("/groups", response_model=list[GroupOut])
def list_groups(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[GroupOut]:
    rows = db.scalars(select(LivestockGroup).order_by(LivestockGroup.created_at.desc())).all()
    return [GroupOut.model_validate(g) for g in rows]


@router.post("/groups", response_model=GroupOut, status_code=201)
def create_group(body: GroupCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> GroupOut:
    _enforce_rules(body, db)
    g = LivestockGroup(**body.model_dump())
    db.add(g)
    db.commit()
    db.refresh(g)
    return GroupOut.model_validate(g)


@router.get("/groups/{group_id}", response_model=GroupOut)
def get_group(group_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> GroupOut:
    g = db.get(LivestockGroup, group_id)
    if not g:
        raise HTTPException(status_code=404, detail="گروه دامی یافت نشد")
    return GroupOut.model_validate(g)


@router.get("/groups/{group_id}/inputs", response_model=list[InputOut])
def list_inputs(group_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[InputOut]:
    rows = db.scalars(
        select(LivestockInput).where(LivestockInput.group_id == group_id).order_by(LivestockInput.date.desc())
    ).all()
    return [InputOut.model_validate(r) for r in rows]


@router.post("/groups/{group_id}/inputs", response_model=InputOut, status_code=201)
def add_input(group_id: UUID, body: InputCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> InputOut:
    if not db.get(LivestockGroup, group_id):
        raise HTTPException(status_code=404, detail="گروه دامی یافت نشد")

    total = body.quantity * body.unit_price
    inp = LivestockInput(group_id=group_id, total_price=total, **body.model_dump())
    db.add(inp)
    db.commit()
    db.refresh(inp)

    livestock_group_cost(db, group_id)
    db.commit()

    return InputOut.model_validate(inp)


@router.get("/groups/{group_id}/production", response_model=list[ProductionOut])
def list_production(group_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[ProductionOut]:
    rows = db.scalars(
        select(LivestockDailyProduction)
        .where(LivestockDailyProduction.group_id == group_id)
        .order_by(LivestockDailyProduction.date.desc())
        .limit(60)
    ).all()
    return [ProductionOut.model_validate(r) for r in rows]


@router.post("/groups/{group_id}/production", response_model=ProductionOut, status_code=201)
def add_production(
    group_id: UUID,
    body: ProductionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ProductionOut:
    if not db.get(LivestockGroup, group_id):
        raise HTTPException(status_code=404, detail="گروه دامی یافت نشد")

    row = LivestockDailyProduction(group_id=group_id, created_by=user.id, **body.model_dump())
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="رکورد تولید این روز قبلاً ثبت شده است")
    db.refresh(row)
    return ProductionOut.model_validate(row)


@router.get("/groups/{group_id}/cost")
def group_cost(group_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> dict:
    try:
        result = livestock_group_cost(db, group_id)
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
