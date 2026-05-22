"""
Seed initial data:
- 1 organization
- 1 central farm, 2 factories, 1 greenhouse
- 5 demo users (one per role)
- Crop types, market prices, processing lines
"""
from datetime import date, timedelta
from sqlalchemy import select

from app.database import SessionLocal
from app.config import settings
from app.core.security import hash_password
from app.models.user import User, UserRole
from app.models.org import Organization, Farm
from app.models.crop import CropType, LandBlock
from app.models.livestock import LivestockGroup
from app.models.processing import ProcessingLine
from app.models.market import MarketPrice
from app.models.inventory import InventoryItem
from app.models.alert import Alert


DEMO_USERS = [
    ("admin@agrocore.local", "مدیر کل", UserRole.admin, "admin1234"),
    ("manager@agrocore.local", "مدیر اجرایی", UserRole.manager, "manager1234"),
    ("farm@agrocore.local", "مدیر مزرعه", UserRole.farm, "farm1234"),
    ("livestock@agrocore.local", "مدیر دامداری", UserRole.livestock, "livestock1234"),
    ("kitchen@agrocore.local", "سرآشپز", UserRole.kitchen, "kitchen1234"),
]


CROP_TYPES = [
    ("tomato", "گوجه فرنگی", "vegetable", 110),
    ("cucumber", "خیار", "vegetable", 70),
    ("pepper", "فلفل دلمه", "vegetable", 90),
    ("eggplant", "بادمجان", "vegetable", 100),
    ("potato", "سیب‌زمینی", "vegetable", 120),
    ("onion", "پیاز", "vegetable", 110),
    ("carrot", "هویج", "vegetable", 90),
    ("wheat", "گندم", "grain", 240),
    ("alfalfa", "یونجه", "fodder", 0),
    ("apple", "سیب", "fruit", 0),
    ("herbs", "سبزی خوردن", "herb", 45),
]


PROCESSING_LINES = [
    ("خط رب گوجه", "گوجه تازه", "رب گوجه ۲x", 10.0, 800, 0.45, 4.0),
    ("خط خیارشور", "خیار", "خیارشور", 1.2, 600, 0.08, 2.5),
    ("خط لبنیات — پنیر", "شیر خام", "پنیر سفید", 10.0, 200, 0.30, 5.0),
    ("خط لبنیات — ماست", "شیر خام", "ماست", 1.0, 500, 0.12, 1.5),
    ("خط خشک کردن سبزی", "سبزی تازه", "سبزی خشک", 8.0, 100, 1.20, 1.0),
]


MARKET_PRICES = [
    ("گوجه فرنگی", "kg", 3_800),
    ("خیار", "kg", 4_200),
    ("سیب‌زمینی", "kg", 2_200),
    ("پیاز", "kg", 1_800),
    ("هویج", "kg", 3_200),
    ("شیر خام", "L", 15_000),
    ("تخم‌مرغ", "each", 8_000),
    ("گوشت گوسفند", "kg", 720_000),
    ("گوشت مرغ", "kg", 145_000),
    ("ماهی قزل‌آلا", "kg", 190_000),
    ("رب گوجه", "kg", 65_000),
    ("پنیر سفید", "kg", 180_000),
    ("ماست", "kg", 35_000),
    ("برنج", "kg", 85_000),
]


INVENTORY_SEED = [
    ("گوجه تازه", "raw_material", "kg", 0, 100, 3_800),
    ("خیار تازه", "raw_material", "kg", 0, 50, 4_200),
    ("شیر خام", "raw_material", "L", 0, 200, 15_000),
    ("تخم‌مرغ", "raw_material", "each", 0, 500, 8_000),
    ("برنج", "raw_material", "kg", 500, 200, 85_000),
    ("روغن", "raw_material", "L", 200, 80, 95_000),
    ("نمک", "raw_material", "kg", 100, 30, 18_000),
    ("کنسانتره دامی ۱۸٪", "feed", "kg", 2_000, 500, 22_000),
    ("کاه و علوفه", "feed", "kg", 5_000, 1_000, 8_000),
    ("بذر گوجه F1", "consumable", "kg", 5, 1, 2_800_000),
    ("کود اوره", "consumable", "kg", 500, 100, 28_000),
    ("کود پتاس", "consumable", "kg", 200, 50, 42_000),
    ("سم آفت‌کش عمومی", "consumable", "L", 30, 10, 380_000),
    ("قوطی فلزی ۷۰۰گرمی", "packaging", "each", 1_000, 200, 3_500),
]


def seed():
    if not settings.seed_on_start:
        return

    db = SessionLocal()
    try:
        if db.scalar(select(User).limit(1)):
            print("[seed] داده‌های اولیه از قبل موجود — رد می‌شود.")
            return

        print("[seed] در حال درج داده‌های اولیه...")

        org = Organization(name="کارخانه‌جات شیشه قزوین", settings={"city": "قزوین", "currency": "تومان"})
        db.add(org)
        db.flush()

        for email, name, role, pwd in DEMO_USERS:
            db.add(User(email=email, name=name, role=role, hashed_password=hash_password(pwd)))

        farms = [
            Farm(org_id=org.id, name="مزرعه مرکزی سالم", type="farm", gps_lat=36.2350, gps_lng=50.1100, area_ha=23.0),
            Farm(org_id=org.id, name="گلخانه سالم", type="greenhouse", gps_lat=36.2340, gps_lng=50.1120, area_ha=2.5),
            Farm(org_id=org.id, name="کارخانه فلوت", type="factory", gps_lat=36.2520, gps_lng=50.0820),
            Farm(org_id=org.id, name="کارخانه مشجر", type="factory", gps_lat=36.2480, gps_lng=50.0900),
        ]
        for f in farms:
            db.add(f)
        db.flush()

        central = farms[0]
        for i, (name, area) in enumerate([("بلوک ۱ — گوجه", 3.0), ("بلوک ۲ — خیار", 2.0), ("بلوک ۳ — یونجه", 5.0), ("بلوک ۴ — گندم", 8.0)], start=1):
            db.add(LandBlock(
                farm_id=central.id, name=name, area_ha=area,
                soil_type="رسی-شنی", irrigation_type="تیپ",
                gps_polygon={"coords": [[36.2380, 50.1050], [36.2380, 50.1200], [36.2300, 50.1200], [36.2300, 50.1050]]},
            ))

        for name_en, name_fa, cat, days in CROP_TYPES:
            db.add(CropType(name=name_en, name_fa=name_fa, category=cat, cycle_days=days))

        db.add_all([
            LivestockGroup(farm_id=central.id, species="cow", purpose="dairy", name="گله گاو شیری", current_count=45, target_count=60),
            LivestockGroup(farm_id=central.id, species="sheep", purpose="meat", name="گله گوسفند", current_count=180, target_count=200),
            LivestockGroup(farm_id=central.id, species="chicken", purpose="egg", name="مرغداری تخم‌گذار", current_count=2000, target_count=2500),
            LivestockGroup(farm_id=central.id, species="chicken", purpose="meat", name="مرغداری گوشتی", current_count=3000, target_count=3000),
            LivestockGroup(farm_id=central.id, species="trout", purpose="meat", name="استخر قزل‌آلا", current_count=5000, target_count=8000),
        ])

        for name, ip, op, ratio, cap, kwh, water in PROCESSING_LINES:
            db.add(ProcessingLine(
                org_id=org.id, name=name, input_product=ip, output_product=op,
                conversion_ratio=ratio, capacity_kg_day=cap,
                electricity_kwh_per_kg=kwh, water_l_per_kg=water,
            ))

        today = date.today()
        for product, unit, price in MARKET_PRICES:
            db.add(MarketPrice(product_name=product, unit=unit, price=price, date=today, source="میدان بار قزوین"))

        for name, cat, unit, qty, min_qty, price in INVENTORY_SEED:
            db.add(InventoryItem(name=name, category=cat, unit=unit, current_qty=qty, min_qty=min_qty, last_unit_price=price))

        db.add_all([
            Alert(severity="info", section="system", title="خوش آمدید", message="سامانه AgroCore OS با موفقیت راه‌اندازی شد."),
            Alert(severity="warning", section="inventory", title="موجودی کم", message="موجودی بذر گوجه F1 نزدیک حداقل است."),
            Alert(severity="warning", section="kitchen", title="یادآوری ۴ وعده ماهی", message="در ماه جاری وعده ماهی برنامه‌ریزی نشده است."),
        ])

        db.commit()
        print("[seed] داده‌های اولیه با موفقیت درج شد.")
        print("[seed] حساب‌های دموی ساخته شد:")
        for email, _, role, pwd in DEMO_USERS:
            print(f"  {role.value:10s}  {email:30s}  رمز: {pwd}")
    except Exception as e:
        db.rollback()
        print(f"[seed] خطا: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
