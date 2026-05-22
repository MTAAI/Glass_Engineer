from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.core.cost_engine import crop_cycle_cost, livestock_group_cost
from app.models.worker import Worker, LaborRecord


router = APIRouter(prefix="/api/v1/workers", tags=["workers"])


class WorkerOut(BaseModel):
    id: UUID
    farm_id: UUID | None = None
    name: str
    code: str | None = None
    type: str
    daily_wage: float
    phone: str | None = None
    active: bool

    model_config = {"from_attributes": True}


class WorkerCreate(BaseModel):
    farm_id: UUID | None = None
    name: str
    code: str | None = None
    type: str = "daily"
    daily_wage: float
    phone: str | None = None


class LaborOut(BaseModel):
    id: UUID
    worker_id: UUID
    date: date
    section: str
    cycle_id: UUID | None = None
    livestock_group_id: UUID | None = None
    hours: float
    task: str | None = None
    daily_wage: float
    total_wage: float
    notes: str | None = None

    model_config = {"from_attributes": True}


class LaborCreate(BaseModel):
    worker_id: UUID
    date: date
    section: str
    cycle_id: UUID | None = None
    livestock_group_id: UUID | None = None
    hours: float = 8.0
    task: str | None = None
    notes: str | None = None


@router.get("", response_model=list[WorkerOut])
def list_workers(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[WorkerOut]:
    rows = db.scalars(select(Worker).where(Worker.active.is_(True))).all()
    return [WorkerOut.model_validate(w) for w in rows]


@router.post("", response_model=WorkerOut, status_code=201)
def create_worker(body: WorkerCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> WorkerOut:
    w = Worker(**body.model_dump())
    db.add(w)
    db.commit()
    db.refresh(w)
    return WorkerOut.model_validate(w)


@router.get("/labor", response_model=list[LaborOut])
def list_labor(
    section: str | None = None,
    cycle_id: UUID | None = None,
    livestock_group_id: UUID | None = None,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
) -> list[LaborOut]:
    stmt = select(LaborRecord).order_by(LaborRecord.date.desc()).limit(200)
    if section:
        stmt = stmt.where(LaborRecord.section == section)
    if cycle_id:
        stmt = stmt.where(LaborRecord.cycle_id == cycle_id)
    if livestock_group_id:
        stmt = stmt.where(LaborRecord.livestock_group_id == livestock_group_id)
    return [LaborOut.model_validate(r) for r in db.scalars(stmt).all()]


@router.post("/labor", response_model=LaborOut, status_code=201)
def add_labor(body: LaborCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> LaborOut:
    worker = db.get(Worker, body.worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="کارگر یافت نشد")

    daily_wage = float(worker.daily_wage)
    total_wage = (daily_wage / 8.0) * body.hours

    record = LaborRecord(
        daily_wage=daily_wage,
        total_wage=total_wage,
        **body.model_dump(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    if body.cycle_id:
        crop_cycle_cost(db, body.cycle_id)
        db.commit()
    if body.livestock_group_id:
        livestock_group_cost(db, body.livestock_group_id)
        db.commit()

    return LaborOut.model_validate(record)
