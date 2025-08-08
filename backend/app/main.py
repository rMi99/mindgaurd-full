from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health, profile, predictions, predict, assessment, dashboard

app = FastAPI(title="MindGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(profile.router)
app.include_router(predictions.router)
app.include_router(predict.router)
app.include_router(assessment.router)
app.include_router(dashboard.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

