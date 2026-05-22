from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.models.crop import LandBlock


router = APIRouter(prefix="/api/v1/blocks", tags=["blocks"])


class BlockOut(BaseModel):
    id: UUID
    farm_id: UUID
    name: str
    area_ha: float
    soil_type: str | None = None
    irrigation_type: str | None = None
    gps_polygon: dict | None = None
    notes: str | None = None
    active: bool

    model_config = {"from_attributes": True}


class BlockCreate(BaseModel):
    farm_id: UUID
    name: str
    area_ha: float
    soil_type: str | None = None
    irrigation_type: str | None = None
    gps_polygon: dict | None = None
    notes: str | None = None


@router.get("", response_model=list[BlockOut])
def list_blocks(farm_id: UUID | None = None, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[BlockOut]:
    stmt = select(LandBlock).where(LandBlock.active.is_(True))
    if farm_id:
        stmt = stmt.where(LandBlock.farm_id == farm_id)
    return [BlockOut.model_validate(b) for b in db.scalars(stmt).all()]


@router.post("", response_model=BlockOut, status_code=201)
def create_block(body: BlockCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> BlockOut:
    block = LandBlock(**body.model_dump())
    db.add(block)
    db.commit()
    db.refresh(block)
    return BlockOut.model_validate(block)


@router.get("/{block_id}", response_model=BlockOut)
def get_block(block_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> BlockOut:
    b = db.get(LandBlock, block_id)
    if not b:
        raise HTTPException(status_code=404, detail="بلوک یافت نشد")
    return BlockOut.model_validate(b)


@router.delete("/{block_id}", status_code=204)
def delete_block(block_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> None:
    b = db.get(LandBlock, block_id)
    if not b:
        raise HTTPException(status_code=404, detail="بلوک یافت نشد")
    b.active = False
    db.commit()
