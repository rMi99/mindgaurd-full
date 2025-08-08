from fastapi import APIRouter
from app.schemas import DashboardGetResponse, DashboardPostRequest, DashboardPostResponse

router = APIRouter(prefix="/api/dashboard")

@router.get("/", response_model=DashboardGetResponse)
async def get_dashboard(userId: str):
    return DashboardGetResponse(history=[], trends={'overallTrend':'stable','insights':[],'recommendations':[]})

@router.post("/", response_model=DashboardPostResponse)
async def post_dashboard(req: DashboardPostRequest):
    return DashboardPostResponse(success=True)
