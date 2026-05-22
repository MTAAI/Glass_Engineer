# AgroCore OS

سامانه جامع مدیریت کشت و صنعت — کارخانه‌جات شیشه قزوین — نسخه ۱.۰

ردیابی کامل از خرید بذر تا سرو غذا در آشپزخانه، با محاسبه قیمت تمام‌شده واقعی هر محصول.

---

## وضعیت توسعه

**فاز ۰ — پایه‌گذاری: ✅ تکمیل شد**

| فاز | عنوان | وضعیت |
|----|------|------|
| ۰ | Docker، Auth، Layout، Migrations | ✅ |
| ۱ | هسته مزرعه (کشت، نهاده، ماشین‌آلات، کارگر) | ⏳ |
| ۲ | دامداری + آبزی‌پروری | ⏳ |
| ۳ | فراوری | ⏳ |
| ۴ | انبار + زنجیره تأمین | ⏳ |
| ۵ | آشپزخانه + HACCP | ⏳ |
| ۶ | مالی و گزارش‌ها | ⏳ |
| ۷ | دوربین، GPS، QR، تکمیل | ⏳ |

---

## شروع سریع

### پیش‌نیاز

- Docker و Docker Compose
- پورت‌های آزاد: 5173 (frontend), 8000 (backend), 5432 (postgres), 6379 (redis), 9000/9001 (minio)

### راه‌اندازی

```bash
cd agrocore
cp .env.example .env
docker compose up -d
```

سپس:

- **UI**: http://localhost:5173
- **API + Swagger**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (agrocore / agrocore_secret)

### حساب‌های دموی

سیستم در اولین اجرا کاربران زیر را ایجاد می‌کند:

| نقش | ایمیل | رمز |
|------|--------|-----|
| مدیر کل | `admin@agrocore.local` | `admin1234` |
| مدیر اجرایی | `manager@agrocore.local` | `manager1234` |
| مدیر مزرعه | `farm@agrocore.local` | `farm1234` |
| مدیر دامداری | `livestock@agrocore.local` | `livestock1234` |
| سرآشپز | `kitchen@agrocore.local` | `kitchen1234` |

---

## معماری

```
React 18 + TypeScript + Vite + Tailwind + shadcn-style UI
                ↓ REST (axios)
FastAPI + Pydantic v2 + JWT + 5 roles
                ↓ SQLAlchemy 2
PostgreSQL 15  │  Redis 7  │  MinIO (S3)
```

### پشته فناوری

**Backend** (پورت 8000)
- FastAPI 0.110 + Uvicorn
- SQLAlchemy 2 + Alembic
- python-jose (JWT) + passlib/bcrypt
- psycopg 3 + Redis 5

**Frontend** (پورت 5173)
- React 18 + TypeScript + Vite 5
- TanStack Query 5 + Zustand (auth store)
- Tailwind CSS 3 (RTL + Vazirmatn)
- React Hook Form + Zod
- Sonner (toast/notification)
- Recharts (در فازهای بعدی) + Leaflet (در فاز ۷)

---

## ساختار پروژه

```
agrocore/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/20260522_0001_init.py
│   └── app/
│       ├── main.py             ← FastAPI app
│       ├── config.py           ← Pydantic Settings
│       ├── database.py         ← SQLAlchemy engine + Base
│       ├── core/
│       │   ├── security.py     ← JWT + bcrypt
│       │   └── deps.py         ← get_current_user + require_roles
│       ├── models/             ← همه ۱۴+ مدل SQLAlchemy
│       ├── schemas/            ← Pydantic DTOs
│       ├── routers/            ← auth, health, dashboard, farms, alerts
│       └── seed/seed_data.py   ← داده‌های اولیه (auto-runs on startup)
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js      ← فونت Vazirmatn + رنگ‌های ماژول‌ها
    └── src/
        ├── main.tsx            ← Bootstrap + QueryClient + Toaster
        ├── App.tsx             ← Router با ۱۶ مسیر
        ├── index.css           ← Tailwind + RTL + کلاس‌های پایه
        ├── components/
        │   ├── Layout.tsx
        │   ├── Sidebar.tsx     ← ۱۴ آیتم منو با گروه‌بندی
        │   ├── Topbar.tsx      ← جستجو، GPS، دوربین، هشدارها
        │   ├── ProtectedRoute.tsx
        │   ├── PageHeader.tsx
        │   └── ComingSoon.tsx
        ├── pages/
        │   ├── Login.tsx       ← فرم کامل با Zod + Demo accounts
        │   ├── Dashboard.tsx   ← KPI زنده از API
        │   └── placeholders.tsx ← ۱۵ صفحه placeholder
        ├── store/auth.ts       ← Zustand + persist
        └── lib/
            ├── api.ts          ← axios + interceptors + types
            └── utils.ts        ← cn(), fa(), tomans()
```

---

## مدل داده — ۱۴ جدول اصلی

- **users** (با ۵ نقش)
- **organizations**, **farms** (مزرعه / گلخانه / کارخانه / انبار)
- **land_blocks**, **crop_types**, **crop_cycles**, **crop_inputs**
- **livestock_groups**, **livestock_inputs**, **livestock_daily_production**, **livestock_events**
- **machinery**, **machinery_usage**
- **workers**, **labor_records**
- **processing_lines**, **processing_batches**
- **inventory_items**, **inventory_transactions**
- **recipes**, **recipe_ingredients**, **menu_entries**, **haccp_records**
- **market_prices**, **alerts**

---

## API — Endpoints فاز ۰

| Method | Endpoint | Auth | شرح |
|--------|----------|------|----|
| GET | `/api/v1/health` | Public | سلامت سامانه + اتصال DB |
| POST | `/api/v1/auth/register` | Public | ثبت‌نام کاربر |
| POST | `/api/v1/auth/login` | Public | ورود + دریافت JWT |
| POST | `/api/v1/auth/token` | Public | OAuth2-compatible برای Swagger |
| GET | `/api/v1/auth/me` | JWT | پروفایل کاربر فعلی |
| GET | `/api/v1/dashboard/kpis` | JWT | KPI‌های زنده داشبورد |
| GET | `/api/v1/farms` | JWT | فهرست مزارع |
| POST | `/api/v1/farms` | JWT | ساخت مزرعه |
| GET | `/api/v1/farms/{id}` | JWT | جزئیات مزرعه |
| GET | `/api/v1/alerts` | JWT | فهرست هشدارها |
| POST | `/api/v1/alerts/{id}/read` | JWT | علامت‌گذاری هشدار خوانده‌شده |
| POST | `/api/v1/alerts/read-all` | JWT | علامت‌گذاری همه |

مستندات کامل: http://localhost:8000/docs

---

## قوانین کسب‌وکار (پیاده‌سازی در فازهای آینده)

- 🐟 قزل‌آلا در تیر، مرداد و شهریور ممنوع
- 🐑 حداکثر تعداد گوسفند: ۲۰۰ رأس
- 🐟 حداقل ۴ وعده ماهی در ماه برای آشپزخانه الزامی
- 🍗 دمای پخت مرغ: حداقل ۷۵ درجه سانتی‌گراد
- 💰 قیمت مواد خودی برابر با قیمت تمام‌شده تولید است، نه قیمت بازار

---

## توسعه محلی (بدون Docker)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg://agrocore:agrocore_secret@localhost:5432/agrocore
alembic upgrade head
python -m app.seed.seed_data
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

---

نسخه ۱.۰ · کارخانه‌جات شیشه قزوین · ۱۴۰۵
