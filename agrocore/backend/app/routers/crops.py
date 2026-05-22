from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.core.cost_engine import crop_cycle_cost
from app.models.crop import CropType, CropCycle, CropInput, LandBlock
from app.models.user import User


router = APIRouter(prefix="/api/v1/crops", tags=["crops"])


class CropTypeOut(BaseModel):
    id: UUID
    name: str
    name_fa: str
    category: str | None = None
    cycle_days: int | None = None

    model_config = {"from_attributes": True}


class CycleOut(BaseModel):
    id: UUID
    block_id: UUID
    block_name: str | None = None
    crop_type_id: UUID
    crop_name: str | None = None
    season: str
    year: int
    plant_date: date | None = None
    expected_harvest_date: date | None = None
    actual_harvest_date: date | None = None
    target_yield_kg: float | None = None
    actual_yield_kg: float | None = None
    status: str
    total_cost_tomans: float | None = None
    cost_per_kg: float | None = None
    notes: str | None = None


class CycleCreate(BaseModel):
    block_id: UUID
    crop_type_id: UUID
    season: str
    year: int
    plant_date: date | None = None
    expected_harvest_date: date | None = None
    target_yield_kg: float | None = None
    notes: str | None = None


class CycleUpdate(BaseModel):
    actual_harvest_date: date | None = None
    actual_yield_kg: float | None = None
    status: str | None = None
    notes: str | None = None


class InputOut(BaseModel):
    id: UUID
    cycle_id: UUID
    input_category: str
    input_type: str
    date: date
    quantity: float
    unit: str
    unit_price: float
    total_price: float
    supplier: str | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class InputCreate(BaseModel):
    input_category: str
    input_type: str
    date: date
    quantity: float
    unit: str
    unit_price: float
    supplier: str | None = None
    notes: str | None = None


def _cycle_to_out(c: CropCycle) -> CycleOut:
    return CycleOut(
        id=c.id, block_id=c.block_id, block_name=c.block.name if c.block else None,
        crop_type_id=c.crop_type_id, crop_name=c.crop_type.name_fa if c.crop_type else None,
        season=c.season, year=c.year,
        plant_date=c.plant_date, expected_harvest_date=c.expected_harvest_date,
        actual_harvest_date=c.actual_harvest_date,
        target_yield_kg=float(c.target_yield_kg) if c.target_yield_kg else None,
        actual_yield_kg=float(c.actual_yield_kg) if c.actual_yield_kg else None,
        status=c.status,
        total_cost_tomans=float(c.total_cost_tomans) if c.total_cost_tomans else None,
        cost_per_kg=float(c.cost_per_kg) if c.cost_per_kg else None,
        notes=c.notes,
    )


@router.get("/types", response_model=list[CropTypeOut])
def list_crop_types(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[CropTypeOut]:
    return [CropTypeOut.model_validate(t) for t in db.scalars(select(CropType).order_by(CropType.name_fa)).all()]


@router.get("/cycles", response_model=list[CycleOut])
def list_cycles(
    block_id: UUID | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[CycleOut]:
    stmt = select(CropCycle).options(joinedload(CropCycle.block), joinedload(CropCycle.crop_type)).order_by(CropCycle.created_at.desc())
    if block_id:
        stmt = stmt.where(CropCycle.block_id == block_id)
    if status:
        stmt = stmt.where(CropCycle.status == status)
    return [_cycle_to_out(c) for c in db.scalars(stmt).all()]


@router.post("/cycles", response_model=CycleOut, status_code=201)
def create_cycle(body: CycleCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CycleOut:
    if not db.get(LandBlock, body.block_id):
        raise HTTPException(status_code=404, detail="بلوک یافت نشد")
    if not db.get(CropType, body.crop_type_id):
        raise HTTPException(status_code=404, detail="نوع محصول یافت نشد")

    cycle = CropCycle(**body.model_dump(), created_by=user.id, status="planned")
    db.add(cycle)
    db.commit()
    db.refresh(cycle, ["block", "crop_type"])
    return _cycle_to_out(cycle)


@router.get("/cycles/{cycle_id}", response_model=CycleOut)
def get_cycle(cycle_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> CycleOut:
    c = db.scalar(
        select(CropCycle).options(joinedload(CropCycle.block), joinedload(CropCycle.crop_type))
        .where(CropCycle.id == cycle_id)
    )
    if not c:
        raise HTTPException(status_code=404, detail="چرخه کشت یافت نشد")
    return _cycle_to_out(c)


@router.patch("/cycles/{cycle_id}", response_model=CycleOut)
def update_cycle(cycle_id: UUID, body: CycleUpdate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> CycleOut:
    c = db.get(CropCycle, cycle_id)
    if not c:
        raise HTTPException(status_code=404, detail="چرخه کشت یافت نشد")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c, ["block", "crop_type"])

    crop_cycle_cost(db, c.id)
    db.commit()
    db.refresh(c)
    return _cycle_to_out(c)


@router.delete("/cycles/{cycle_id}", status_code=204)
def delete_cycle(cycle_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> None:
    c = db.get(CropCycle, cycle_id)
    if not c:
        raise HTTPException(status_code=404, detail="چرخه کشت یافت نشد")
    db.delete(c)
    db.commit()


@router.get("/cycles/{cycle_id}/inputs", response_model=list[InputOut])
def list_inputs(cycle_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[InputOut]:
    rows = db.scalars(select(CropInput).where(CropInput.cycle_id == cycle_id).order_by(CropInput.date.desc())).all()
    return [InputOut.model_validate(r) for r in rows]


@router.post("/cycles/{cycle_id}/inputs", response_model=InputOut, status_code=201)
def add_input(cycle_id: UUID, body: InputCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> InputOut:
    if not db.get(CropCycle, cycle_id):
        raise HTTPException(status_code=404, detail="چرخه کشت یافت نشد")

    total = body.quantity * body.unit_price
    inp = CropInput(
        cycle_id=cycle_id,
        total_price=total,
        created_by=user.id,
        **body.model_dump(),
    )
    db.add(inp)
    db.commit()
    db.refresh(inp)

    crop_cycle_cost(db, cycle_id)
    db.commit()

    return InputOut.model_validate(inp)


@router.delete("/cycles/{cycle_id}/inputs/{input_id}", status_code=204)
def delete_input(cycle_id: UUID, input_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> None:
    inp = db.get(CropInput, input_id)
    if not inp or inp.cycle_id != cycle_id:
        raise HTTPException(status_code=404, detail="نهاده یافت نشد")
    db.delete(inp)
    db.commit()
    crop_cycle_cost(db, cycle_id)
    db.commit()


@router.get("/cycles/{cycle_id}/cost")
def cycle_cost(cycle_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> dict:
    try:
        result = crop_cycle_cost(db, cycle_id)
        db.commit()
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
