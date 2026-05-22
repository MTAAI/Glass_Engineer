from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.deps import get_current_user
from app.models.alert import Alert
from app.schemas.common import AlertOut


router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut])
def list_alerts(unread_only: bool = False, db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[AlertOut]:
    stmt = select(Alert).order_by(Alert.created_at.desc()).limit(50)
    if unread_only:
        stmt = stmt.where(Alert.read.is_(False))
    return [AlertOut.model_validate(a) for a in db.scalars(stmt).all()]


@router.post("/{alert_id}/read", status_code=204)
def mark_read(alert_id: UUID, db: Session = Depends(get_db), _=Depends(get_current_user)) -> None:
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="هشدار یافت نشد")
    alert.read = True
    db.commit()


@router.post("/read-all", status_code=204)
def mark_all_read(db: Session = Depends(get_db), _=Depends(get_current_user)) -> None:
    db.execute(update(Alert).where(Alert.read.is_(False)).values(read=True))
    db.commit()
