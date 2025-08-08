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
