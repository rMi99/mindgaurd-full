#!/usr/bin/env bash
set -e

echo "🛠  Setting up MindGuard backend..."

# 1️⃣ Create directories
mkdir -p app/models app/services app/routes scripts data

# 2️⃣ Write app/models/model.py
cat > app/models/model.py << 'EOF'
import torch
import torch.nn as nn

class RiskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super(RiskModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
EOF

# 3️⃣ Write app/services/model_service.py
cat > app/services/model_service.py << 'EOF'
import pickle
import torch
from typing import Tuple, List
from app.models.model import RiskModel

_model: RiskModel = None
_scaler = None
_risk_map = {0: "low", 1: "moderate", 2: "high"}

def load_model_and_scaler(
    model_path: str = "./app/models/checkpoint.pt", 
    scaler_path: str = "./app/models/checkpoint_scaler.pkl"
):
    global _model, _scaler
    with open(scaler_path, "rb") as f:
        _scaler = pickle.load(f)

    input_dim = _scaler.mean_.shape[0]
    output_dim = len(_risk_map)
    _model = RiskModel(input_dim, hidden_dim=64, output_dim=output_dim)
    _model.load_state_dict(torch.load(model_path))
    _model.eval()

def predict_risk(features: List[float]) -> Tuple[str, float]:
    import numpy as np
    x = _scaler.transform([features])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = _model(x_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(probs.argmax())
    return _risk_map[idx], float(probs[idx])
EOF

# 4️⃣ Write app/schemas.py
cat > app/schemas.py << 'EOF'
from pydantic import BaseModel
from typing import List, Literal, Dict, Any

class PyTorchAnalysisRequest(BaseModel):
    phq9: Dict[str, List[int]]
    sleep: Dict[str, Any]
    behavioral: Dict[str, Any]

class KeyFactor(BaseModel):
    value: str
    impact: Literal['low','moderate','high']

class Intervention(BaseModel):
    type: str
    duration: str = None
    plan: str = None
    reason: str = None
    title: str
    description: str

class PyTorchAnalysisResponse(BaseModel):
    risk_level: Literal['low','moderate','high']
    confidence: float
    key_factors: Dict[str, KeyFactor]
    interventions: Dict[Literal['immediate','longterm'], List[Intervention]]
    biometric_scores: Dict[str, float]
    recommendations: List[str]

class DashboardEntry(BaseModel):
    date: str
    phq9Score: int
    riskLevel: Literal['low','moderate','high']
    sleepHours: str
    sleepQuality: str
    sleepHoursNumeric: float
    stressLevel: str
    exerciseFrequency: str
    socialSupport: str

class DashboardTrends(BaseModel):
    overallTrend: Literal['improving','worsening','stable']
    insights: List[str]
    recommendations: List[str]

class DashboardGetResponse(BaseModel):
    history: List[DashboardEntry]
    trends: DashboardTrends

class DashboardPostRequest(BaseModel):
    userId: str
    assessmentData: Dict[str, Any]

class DashboardPostResponse(BaseModel):
    success: bool

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    score: float

class Demographics(BaseModel):
    age: str
    gender: str
    region: str
    education: str
    employmentStatus: str

class AssessmentRequest(BaseModel):
    demographics: Demographics
    phq9: Dict[str, List[int]]
    sleep: Dict[str, Any]

class AssessmentResponse(BaseModel):
    riskLevel: Literal['low','moderate','high']
    phq9Score: int
    confidenceScore: float
    riskFactors: List[str]
    protectiveFactors: List[str]
    recommendations: List[str]
    culturalConsiderations: List[str]
    emergencyResources: List[str]
EOF

# 5️⃣ Write app/routes/pytorch_analysis.py
cat > app/routes/pytorch_analysis.py << 'EOF'
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
EOF

# 6️⃣ Write app/routes/dashboard.py
cat > app/routes/dashboard.py << 'EOF'
from fastapi import APIRouter
from app.schemas import DashboardGetResponse, DashboardPostRequest, DashboardPostResponse

router = APIRouter(prefix="/api/dashboard")

@router.get("/", response_model=DashboardGetResponse)
async def get_dashboard(userId: str):
    return DashboardGetResponse(history=[], trends={'overallTrend':'stable','insights':[],'recommendations':[]})

@router.post("/", response_model=DashboardPostResponse)
async def post_dashboard(req: DashboardPostRequest):
    return DashboardPostResponse(success=True)
EOF

# 7️⃣ Write app/routes/sentiment_analysis.py
cat > app/routes/sentiment_analysis.py << 'EOF'
from fastapi import APIRouter
from app.schemas import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/api/sentiment-analysis")

@router.post("/", response_model=SentimentResponse)
async def sentiment_analysis(req: SentimentRequest):
    score = 0.85
    return SentimentResponse(score=score)
EOF

# 8️⃣ Write app/routes/assessment.py
cat > app/routes/assessment.py << 'EOF'
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
EOF

# 9️⃣ Write app/main.py
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from app.routes import pytorch_analysis, dashboard, sentiment_analysis, assessment
from app.services.model_service import load_model_and_scaler

app = FastAPI()
load_model_and_scaler()

app.include_router(pytorch_analysis.router)
app.include_router(dashboard.router)
app.include_router(sentiment_analysis.router)
app.include_router(assessment.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
EOF

# 🔟 Write requirements.txt
cat > requirements.txt << 'EOF'
fastapi
uvicorn[standard]
torch
scikit-learn
python-dotenv
pydantic
EOF

# .env.example
cat > .env.example << 'EOF'
MODEL_PATH=./app/models/checkpoint.pt
EOF

# 1⓪ Create a simple init_db stub if missing
if [ ! -f scripts/init_db.py ]; then
  cat > scripts/init_db.py << 'EOF'
# scripts/init_db.py
# Add your MongoDB initialization logic here
if __name__ == "__main__":
    print("Initializing MongoDB collections and indexes...")
EOF
fi

# 1① Setup virtualenv & install
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1② Optional DB init
python scripts/init_db.py || true

echo
echo "✅ MindGuard backend scaffolded!"
echo "👉 Activate it with:  source venv/bin/activate"
echo "👉 Then run: uvicorn app.main:app --reload"
