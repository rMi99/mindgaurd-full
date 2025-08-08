from fastapi import APIRouter
from app.schemas import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/api/sentiment-analysis")

@router.post("/", response_model=SentimentResponse)
async def sentiment_analysis(req: SentimentRequest):
    score = 0.85
    return SentimentResponse(score=score)
