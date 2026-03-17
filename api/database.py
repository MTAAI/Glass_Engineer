"""
Glass Expert AI — Database shim
Re-exports from api.core.database for backwards compatibility.
Allows Engineer Z's conversations.py to import from api.database.
"""
from api.core.database import (
    get_db_conn,
    return_db_conn,
    get_db,
    check_db_health,
)

__all__ = ["get_db_conn", "return_db_conn", "get_db", "check_db_health"]