from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.models.org import Farm
from app.schemas.common import FarmOut


router = APIRouter(prefix="/api/v1/farms", tags=["farms"])


class FarmCreate(BaseModel):
    name: str
    type: str = "farm"
    gps_lat: float | None = None
    gps_lng: float | None = None
    area_ha: float | None = None
    notes: str | None = None


@router.get("", response_model=list[FarmOut])
def list_farms(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[FarmOut]:
    rows = db.scalars(select(Farm).order_by(Farm.created_at.desc())).all()
    return [FarmOut.model_validate(r) for r in rows]


@router.post("", response_model=FarmOut, status_code=201)
def create_farm(body: FarmCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> FarmOut:
    org_id = db.scalar(select(Farm.org_id).limit(1))
    if not org_id:
        from app.models.org import Organization
        org = db.scalar(select(Organization).limit(1))
        if not org:
            org = Organization(name="کشت‌وصنعت")
            db.add(org)
            db.flush()
        org_id = org.id

    farm = Farm(org_id=org_id, **body.model_dump())
    db.add(farm)
    db.commit()
    db.refresh(farm)
    return FarmOut.model_validate(farm)


@router.get("/{farm_id}", response_model=FarmOut)
def get_farm(farm_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> FarmOut:
    farm = db.get(Farm, farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail="مزرعه یافت نشد")
    return FarmOut.model_validate(farm)
