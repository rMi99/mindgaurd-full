from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health, profile, predictions, predict, assessment, dashboard, research, history, admin, auth, global_stats, user_management

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
app.include_router(research.router)
app.include_router(history.router)
app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(global_stats.router)
app.include_router(user_management.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

