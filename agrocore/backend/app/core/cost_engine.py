"""
Cost engine — the heart of AgroCore OS.

Calculates total cost and cost-per-unit for crop cycles, livestock groups,
and processing batches by aggregating all recorded inputs.

Categories used:
- direct_materials: seed, fertilizer, pesticide, packaging, feed, medicine, vet
- direct_labor: labor records linked to the entity
- machinery: machinery usage for crop cycles
- utilities: water, electricity, fuel
- overhead: explicitly tagged overhead inputs
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.crop import CropCycle, CropInput
from app.models.livestock import LivestockGroup, LivestockInput, LivestockDailyProduction
from app.models.machinery import MachineryUsage
from app.models.worker import LaborRecord
from app.models.processing import ProcessingBatch
from app.models.market import MarketPrice


# Category groupings — keep aligned with model `input_category` values.
DIRECT_MATERIAL_CATS = {"seed", "seedling", "fertilizer", "pesticide", "packaging", "feed", "medicine", "vet", "bedding"}
UTILITY_CATS = {"water", "electricity", "fuel"}
OVERHEAD_CATS = {"overhead", "transport", "waste"}


@dataclass
class CostBreakdown:
    direct_materials: float = 0.0
    direct_labor: float = 0.0
    machinery: float = 0.0
    utilities: float = 0.0
    overhead: float = 0.0
    by_subcategory: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.direct_materials
            + self.direct_labor
            + self.machinery
            + self.utilities
            + self.overhead
        )

    def as_dict(self) -> dict:
        return {
            "direct_materials": self.direct_materials,
            "direct_labor": self.direct_labor,
            "machinery": self.machinery,
            "utilities": self.utilities,
            "overhead": self.overhead,
            "total": self.total,
            "by_subcategory": self.by_subcategory,
        }


def _bucket(category: str) -> str:
    c = (category or "").lower()
    if c in DIRECT_MATERIAL_CATS:
        return "direct_materials"
    if c in UTILITY_CATS:
        return "utilities"
    if c in OVERHEAD_CATS:
        return "overhead"
    if c == "labor":
        return "direct_labor"
    if c in {"machinery", "machine"}:
        return "machinery"
    return "overhead"


def _accumulate(breakdown: CostBreakdown, category: str, amount: float) -> None:
    bucket = _bucket(category)
    setattr(breakdown, bucket, getattr(breakdown, bucket) + amount)
    breakdown.by_subcategory[category] = breakdown.by_subcategory.get(category, 0.0) + amount


def get_market_price(db: Session, product_name: str) -> float | None:
    row = db.scalar(
        select(MarketPrice)
        .where(MarketPrice.product_name == product_name)
        .order_by(MarketPrice.date.desc())
        .limit(1)
    )
    return float(row.price) if row else None


def crop_cycle_cost(db: Session, cycle_id: UUID) -> dict:
    cycle = db.get(CropCycle, cycle_id)
    if not cycle:
        raise ValueError("Crop cycle not found")

    breakdown = CostBreakdown()

    # Inputs from the inputs ledger
    inputs: Iterable[CropInput] = db.scalars(select(CropInput).where(CropInput.cycle_id == cycle_id)).all()
    for inp in inputs:
        _accumulate(breakdown, inp.input_category, float(inp.total_price or 0))

    # Labor records linked to this cycle
    labor_total = db.scalar(
        select(func.coalesce(func.sum(LaborRecord.total_wage), 0)).where(LaborRecord.cycle_id == cycle_id)
    ) or 0
    if labor_total:
        _accumulate(breakdown, "labor", float(labor_total))

    # Machinery usage linked to this cycle
    machinery_total = db.scalar(
        select(func.coalesce(func.sum(MachineryUsage.total_cost), 0)).where(MachineryUsage.cycle_id == cycle_id)
    ) or 0
    if machinery_total:
        _accumulate(breakdown, "machinery", float(machinery_total))

    yield_kg = float(cycle.actual_yield_kg or 0)
    cost_per_kg = breakdown.total / yield_kg if yield_kg > 0 else None

    product_name = cycle.crop_type.name_fa if cycle.crop_type else None
    market_price = get_market_price(db, product_name) if product_name else None
    savings_pct = None
    if cost_per_kg is not None and market_price and market_price > 0:
        savings_pct = round((market_price - cost_per_kg) / market_price * 100, 1)

    # Persist the calculated fields back on the cycle
    cycle.total_cost_tomans = breakdown.total
    cycle.cost_per_kg = cost_per_kg
    db.add(cycle)

    return {
        "cycle_id": str(cycle.id),
        "crop": product_name,
        "year": cycle.year,
        "season": cycle.season,
        "status": cycle.status,
        "yield": {
            "target_kg": float(cycle.target_yield_kg) if cycle.target_yield_kg else None,
            "actual_kg": yield_kg,
        },
        "breakdown": breakdown.as_dict(),
        "summary": {
            "total_cost": breakdown.total,
            "cost_per_kg": cost_per_kg,
            "market_price_per_kg": market_price,
            "savings_vs_market_pct": savings_pct,
        },
    }


def livestock_group_cost(db: Session, group_id: UUID) -> dict:
    group = db.get(LivestockGroup, group_id)
    if not group:
        raise ValueError("Livestock group not found")

    breakdown = CostBreakdown()

    inputs = db.scalars(select(LivestockInput).where(LivestockInput.group_id == group_id)).all()
    for inp in inputs:
        _accumulate(breakdown, inp.input_category, float(inp.total_price or 0))

    labor_total = db.scalar(
        select(func.coalesce(func.sum(LaborRecord.total_wage), 0)).where(LaborRecord.livestock_group_id == group_id)
    ) or 0
    if labor_total:
        _accumulate(breakdown, "labor", float(labor_total))

    # Total production (milk in L, eggs in count, weight gain in kg)
    prod_rows = db.execute(
        select(LivestockDailyProduction.production_type, func.coalesce(func.sum(LivestockDailyProduction.amount), 0))
        .where(LivestockDailyProduction.group_id == group_id)
        .group_by(LivestockDailyProduction.production_type)
    ).all()
    production = {ptype: float(amount) for ptype, amount in prod_rows}

    cost_per_unit: dict[str, float] = {}
    if breakdown.total > 0:
        for ptype, amount in production.items():
            if amount > 0:
                cost_per_unit[ptype] = breakdown.total / amount

    group.total_cost_to_date = breakdown.total
    db.add(group)

    return {
        "group_id": str(group.id),
        "name": group.name,
        "species": group.species,
        "purpose": group.purpose,
        "head_count": group.current_count,
        "breakdown": breakdown.as_dict(),
        "production": production,
        "cost_per_unit": cost_per_unit,
    }


def processing_batch_cost(db: Session, batch_id: UUID) -> dict:
    batch = db.get(ProcessingBatch, batch_id)
    if not batch:
        raise ValueError("Processing batch not found")

    total = (
        float(batch.input_cost or 0)
        + float(batch.labor_cost or 0)
        + float(batch.energy_cost or 0)
        + float(batch.packaging_cost or 0)
        + float(batch.other_cost or 0)
    )
    cost_per_kg = total / float(batch.output_qty_kg) if batch.output_qty_kg else None

    batch.total_cost = total
    batch.cost_per_kg = cost_per_kg
    db.add(batch)

    return {
        "batch_id": str(batch.id),
        "date": batch.date.isoformat(),
        "input_qty_kg": float(batch.input_qty_kg),
        "output_qty_kg": float(batch.output_qty_kg or 0),
        "waste_kg": float(batch.waste_kg or 0),
        "breakdown": {
            "input": float(batch.input_cost or 0),
            "labor": float(batch.labor_cost or 0),
            "energy": float(batch.energy_cost or 0),
            "packaging": float(batch.packaging_cost or 0),
            "other": float(batch.other_cost or 0),
            "total": total,
        },
        "cost_per_kg": cost_per_kg,
    }
