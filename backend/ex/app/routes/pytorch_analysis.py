from fastapi import APIRouter
from app.schemas import PyTorchAnalysisRequest, PyTorchAnalysisResponse, KeyFactor, Intervention
from app.services.model_service import predict_risk

router = APIRouter(prefix="/api/pytorch-analysis")

@router.post("/", response_model=PyTorchAnalysisResponse)
async def pytorch_analysis(req: PyTorchAnalysisRequest):
    phq_scores = req.phq9.get('scores', [])
    sleep_hours = req.sleep.get('averageHours', 0)
    stress = req.behavioral.get('stressLevel', 0)
    features = phq_scores + [sleep_hours, stress]

    risk, conf = predict_risk(features)
    total_phq = sum(phq_scores)

    key_factors = {
        'sleep_deficit': KeyFactor(value=f"{sleep_hours}h/night", impact="moderate"),
        'phq9_score': KeyFactor(value=f"{total_phq}/27", impact="moderate")
    }
    interventions = {
        'immediate': [
            Intervention(
                type="breathing", duration="5min", reason="elevated_stress_indicators",
                title="4-4-6 Breathing Exercise",
                description="Controlled breathing to activate parasympathetic nervous system and reduce acute stress"
            )
        ],
        'longterm': [
            Intervention(
                type="sleep_hygiene", plan="7-9_hour_sleep_schedule",
                title="Sleep Optimization Program",
                description="Establish consistent sleep-wake cycle to improve mood regulation and cognitive function"
            )
        ]
    }
    biometric_scores = {
        'sleep': sleep_hours,
        'mood': 0.0,
        'social': float(req.behavioral.get('socialConnections', 0)),
        'stress': float(stress),
        'energy': 0.0
    }
    recommendations = [
        "Prioritize 7-9 hours of sleep nightly for optimal mental health",
        "Practice daily stress-reduction techniques like meditation or deep breathing"
    ]

    return PyTorchAnalysisResponse(
        risk_level=risk,
        confidence=conf,
        key_factors=key_factors,
        interventions=interventions,
        biometric_scores=biometric_scores,
        recommendations=recommendations
    )
