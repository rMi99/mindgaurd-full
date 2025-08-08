from fastapi import APIRouter
from app.schemas import AssessmentRequest, AssessmentResponse
from app.services.model_service import predict_risk

router = APIRouter(prefix="/api/assessment")

@router.post("/", response_model=AssessmentResponse)
async def assessment(req: AssessmentRequest):
    phq_scores = req.phq9.get('scores', [])
    sleep_hours = req.sleep.get('averageHours', 0)
    features = phq_scores + [sleep_hours]
    risk, conf = predict_risk(features)

    return AssessmentResponse(
        riskLevel=risk,
        phq9Score=sum(phq_scores),
        confidenceScore=conf,
        riskFactors=["Elevated PHQ-9 score."],
        protectiveFactors=["Strong social support network."],
        recommendations=["Consider speaking with a healthcare provider about your mental health."],
        culturalConsiderations=["In your region, it may be helpful to connect with community elders for support."],
        emergencyResources=[]
    )
