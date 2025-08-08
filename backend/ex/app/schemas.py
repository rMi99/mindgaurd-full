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
