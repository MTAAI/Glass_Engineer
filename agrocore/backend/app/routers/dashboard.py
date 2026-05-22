from datetime import date, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import get_current_user
from app.models.org import Farm
from app.models.crop import CropCycle
from app.models.livestock import LivestockGroup, LivestockDailyProduction
from app.models.inventory import InventoryItem
from app.models.alert import Alert
from app.models.kitchen import MenuEntry, Recipe
from app.schemas.common import DashboardKPIs


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/kpis", response_model=DashboardKPIs)
def kpis(db: Session = Depends(get_db), _=Depends(get_current_user)) -> DashboardKPIs:
    today = date.today()

    active_farms = db.scalar(select(func.count()).select_from(Farm)) or 0

    active_cycles = db.scalar(
        select(func.count()).select_from(CropCycle).where(CropCycle.status.in_(["planned", "planting", "growing"]))
    ) or 0

    livestock_groups = db.scalar(select(func.count()).select_from(LivestockGroup)) or 0
    livestock_head = db.scalar(select(func.coalesce(func.sum(LivestockGroup.current_count), 0))) or 0

    milk_today = db.scalar(
        select(func.coalesce(func.sum(LivestockDailyProduction.amount), 0)).where(
            and_(LivestockDailyProduction.date == today, LivestockDailyProduction.production_type == "milk")
        )
    ) or 0.0

    eggs_today = db.scalar(
        select(func.coalesce(func.sum(LivestockDailyProduction.amount), 0)).where(
            and_(LivestockDailyProduction.date == today, LivestockDailyProduction.production_type == "egg")
        )
    ) or 0

    inv_items = db.scalar(select(func.count()).select_from(InventoryItem)) or 0
    low_stock = db.scalar(
        select(func.count()).select_from(InventoryItem).where(InventoryItem.current_qty <= InventoryItem.min_qty)
    ) or 0

    open_alerts = db.scalar(select(func.count()).select_from(Alert).where(Alert.read.is_(False))) or 0

    portions_today = db.scalar(
        select(func.coalesce(func.sum(MenuEntry.portion_count), 0)).where(MenuEntry.date == today)
    ) or 0

    avg_cost = db.scalar(
        select(func.avg(Recipe.cost_per_portion)).where(Recipe.cost_per_portion.isnot(None))
    )

    return DashboardKPIs(
        active_farms=int(active_farms),
        active_cycles=int(active_cycles),
        livestock_groups=int(livestock_groups),
        livestock_head=int(livestock_head),
        daily_milk_l=float(milk_today),
        daily_eggs=int(eggs_today),
        inventory_items=int(inv_items),
        low_stock_count=int(low_stock),
        open_alerts=int(open_alerts),
        portions_today=int(portions_today),
        avg_cost_per_portion=float(avg_cost) if avg_cost is not None else None,
    )
