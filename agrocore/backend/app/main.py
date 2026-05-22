from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import auth, health, dashboard, farms, alerts


app = FastAPI(
    title="AgroCore OS",
    description="سامانه جامع مدیریت کشت و صنعت — کارخانه‌جات شیشه قزوین",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(farms.router)
app.include_router(alerts.router)


@app.get("/")
def root() -> dict:
    return {
        "app": "AgroCore OS",
        "version": "1.0.0",
        "docs": "/docs",
    }
