from fastapi import APIRouter, Depends, HTTPException

from app.api.auth_routes import require_admin
from app.core.database import find_user_by_id, list_users, set_admin

admin_router = APIRouter(prefix="/admin")


@admin_router.get("/users")
def get_users(admin: dict = Depends(require_admin)):
    return {
        "users": [
            {
                "id":         u["id"],
                "username":   u["username"],
                "email":      u["email"],
                "is_admin":   bool(u["is_admin"]),
                "created_at": u["created_at"],
            }
            for u in list_users()
        ]
    }


@admin_router.post("/users/{user_id}/promote")
def promote(user_id: int, admin: dict = Depends(require_admin)):
    if not find_user_by_id(user_id):
        raise HTTPException(404, "User not found.")
    set_admin(user_id, True)
    return {"success": True, "user_id": user_id, "is_admin": True}


@admin_router.post("/users/{user_id}/demote")
def demote(user_id: int, admin: dict = Depends(require_admin)):
    if user_id == int(admin["sub"]):
        raise HTTPException(400, "You cannot demote yourself.")
    if not find_user_by_id(user_id):
        raise HTTPException(404, "User not found.")
    set_admin(user_id, False)
    return {"success": True, "user_id": user_id, "is_admin": False}
