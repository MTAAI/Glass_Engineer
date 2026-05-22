from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime


class FarmOut(BaseModel):
    id: UUID
    name: str
    type: str
    gps_lat: float | None = None
    gps_lng: float | None = None
    area_ha: float | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class AlertOut(BaseModel):
    id: UUID
    severity: str
    section: str
    title: str
    message: str
    ref_id: str | None = None
    read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DashboardKPIs(BaseModel):
    active_farms: int
    active_cycles: int
    livestock_groups: int
    livestock_head: int
    daily_milk_l: float
    daily_eggs: int
    inventory_items: int
    low_stock_count: int
    open_alerts: int
    portions_today: int
    avg_cost_per_portion: float | None = None
