from fastapi import APIRouter, HTTPException
from app.services.db import db
from app.schemas.schemas import ProfileResponse

router = APIRouter()

@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    user = await db.users.find_one({"user_id": user_id}, {'_id':0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
