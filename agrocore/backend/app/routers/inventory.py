from datetime import date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.core.deps import get_current_user
from app.models.inventory import InventoryItem, InventoryTransaction
from app.models.user import User


router = APIRouter(prefix="/api/v1/inventory", tags=["inventory"])


class ItemOut(BaseModel):
    id: UUID
    name: str
    category: str
    unit: str
    current_qty: float
    min_qty: float
    last_unit_price: float | None = None
    notes: str | None = None
    low_stock: bool

    model_config = {"from_attributes": True}


class ItemCreate(BaseModel):
    name: str
    category: str
    unit: str
    current_qty: float = 0
    min_qty: float = 0
    last_unit_price: float | None = None
    notes: str | None = None


class TxOut(BaseModel):
    id: UUID
    item_id: UUID
    direction: str
    source: str
    ref_id: str | None = None
    date: date
    quantity: float
    unit_price: float | None = None
    total_price: float | None = None
    batch_code: str | None = None
    expiry_date: date | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class TxCreate(BaseModel):
    direction: str
    source: str
    ref_id: str | None = None
    date: date
    quantity: float
    unit_price: float | None = None
    batch_code: str | None = None
    expiry_date: date | None = None
    notes: str | None = None


def _to_out(item: InventoryItem) -> ItemOut:
    return ItemOut(
        id=item.id, name=item.name, category=item.category, unit=item.unit,
        current_qty=float(item.current_qty), min_qty=float(item.min_qty),
        last_unit_price=float(item.last_unit_price) if item.last_unit_price else None,
        notes=item.notes,
        low_stock=float(item.current_qty) <= float(item.min_qty),
    )


@router.get("/items", response_model=list[ItemOut])
def list_items(low_only: bool = False, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[ItemOut]:
    rows = db.scalars(select(InventoryItem).order_by(InventoryItem.name)).all()
    outs = [_to_out(r) for r in rows]
    if low_only:
        outs = [o for o in outs if o.low_stock]
    return outs


@router.post("/items", response_model=ItemOut, status_code=201)
def create_item(body: ItemCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> ItemOut:
    item = InventoryItem(**body.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return _to_out(item)


@router.post("/items/{item_id}/transactions", response_model=TxOut, status_code=201)
def add_transaction(item_id: UUID, body: TxCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> TxOut:
    item = db.get(InventoryItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="کالا یافت نشد")
    if body.direction not in ("in", "out"):
        raise HTTPException(status_code=400, detail="جهت تراکنش باید in یا out باشد")
    if body.direction == "out" and float(item.current_qty) < body.quantity:
        raise HTTPException(status_code=400, detail="موجودی کافی نیست")

    total = (body.unit_price or 0) * body.quantity if body.unit_price else None

    tx = InventoryTransaction(
        item_id=item_id,
        total_price=total,
        created_by=user.id,
        **body.model_dump(),
    )
    delta = body.quantity if body.direction == "in" else -body.quantity
    item.current_qty = float(item.current_qty) + delta
    if body.direction == "in" and body.unit_price:
        item.last_unit_price = body.unit_price

    db.add_all([tx, item])
    db.commit()
    db.refresh(tx)
    return TxOut.model_validate(tx)


@router.get("/items/{item_id}/transactions", response_model=list[TxOut])
def list_transactions(item_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[TxOut]:
    rows = db.scalars(
        select(InventoryTransaction).where(InventoryTransaction.item_id == item_id).order_by(InventoryTransaction.date.desc()).limit(100)
    ).all()
    return [TxOut.model_validate(r) for r in rows]
