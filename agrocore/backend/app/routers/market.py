from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.models.market import MarketPrice


router = APIRouter(prefix="/api/v1/market", tags=["market"])


class PriceOut(BaseModel):
    id: UUID
    product_name: str
    unit: str
    price: float
    date: date
    source: str | None = None

    model_config = {"from_attributes": True}


@router.get("/prices", response_model=list[PriceOut])
def latest_prices(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[PriceOut]:
    """Latest market price per product."""
    subq = (
        select(MarketPrice.product_name, func.max(MarketPrice.date).label("max_date"))
        .group_by(MarketPrice.product_name)
        .subquery()
    )
    rows = db.scalars(
        select(MarketPrice).join(
            subq,
            (MarketPrice.product_name == subq.c.product_name) & (MarketPrice.date == subq.c.max_date),
        ).order_by(MarketPrice.product_name)
    ).all()
    return [PriceOut.model_validate(r) for r in rows]
