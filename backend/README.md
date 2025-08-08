# MindGuard Backend

AI-powered mental health risk prediction server using FastAPI, MongoDB, and PyTorch.

## Setup

1. Clone repository:
   \`\`\`bash
   git clone https://github.com/rmi99/mindguard-backend.git
   cd mindguard-backend
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Create environment file:
   \`\`\`bash
   cp .env.example .env
   # Edit .env to set MONGODB_URI if needed
   \`\`\`

4. Place your training dataset at \`data/dataset.csv\`. It should have numeric features (e.g., age, phq9_score, sleep_hours) and a \`label\` column (0=low,1=moderate,2=high).

5. Initialize database:
   \`\`\`bash
   python scripts/init_db.py
   \`\`\`

6. Train model:
   \`\`\`bash
   python scripts/train_model.py
   \`\`\`

7. Run server:
   \`\`\`bash
   uvicorn app.main:app --reload
   \`\`\`

## API Endpoints
- \`GET /health\` → \`{ "status": "ok" }\`
- \`GET /profile/{user_id}\`
- \`GET /predictions/{user_id}\`
- \`POST /predict\` → JSON payload matching \`PredictRequest\`

## Frontend Integration
The React/Next.js frontend should call these routes under \`/api\` as documented above. Ensure \`MONGODB_URI\` and \`MODEL_PATH\` are consistent with your \`.env\`.


------------------------------
To achieve ≥80% accuracy, we’ll merge and engineer three complementary public datasets:

OSMI Mental Health in Tech Survey (Kaggle)

URL: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

Key features: age, gender, self_employed, family_history, work_interfere, …, leave, treatment

Target: treatment (0/1)

PHQ-9 Depression Screening (Kaggle)

URL: https://www.kaggle.com/datasets/anjanikathuria/phq-9-depression-dataset

Features: phq9_score, age, gender, label (0=low,1=mod,2=high)

Usage: Map phq9_score to our survey records by matching on demographic bins (age group & gender)

Sleep Health Dataset (Kaggle)

URL: https://www.kaggle.com/datasets/shivamb/sleep-health-dataset

Features: sleep_hours, sleep_quality_score, age, gender

Usage: Merge sleep_hours via demographic bins