# ML Multi-Disease Predictor

A full-stack machine learning web app that predicts the likelihood of **Diabetes** and **Heart Disease** based on user-provided medical inputs. Built with a FastAPI backend serving trained scikit-learn pipelines, and a Streamlit frontend for the user interface.

## Live Demo

- **Frontend (Streamlit):** https://ml-multi-disease-predictor-2.onrender.com
- **Backend API docs (Swagger):** https://ml-multi-disease-predictor-1.onrender.com/docs

## Features

- Predicts diabetes risk using patient health metrics (glucose, BMI, blood pressure, age, etc.)
- Predicts heart disease risk using clinical features (cholesterol, ECG results, chest pain type, etc.)
- REST API backend (FastAPI) decoupled from the UI (Streamlit), independently deployable
- Trained scikit-learn pipelines (preprocessing + model) serialized with `joblib`
- Interactive Swagger/OpenAPI docs auto-generated at `/docs`

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn, XGBoost, joblib |
| Frontend | Streamlit |
| Data | Pandas, NumPy |
| Deployment | Render (two independent web services) |

## Project Structure

```
ML-Multi-Disease-Predictor/
├── dataset/                  # Training datasets (diabetes.csv, heart.csv)
├── model_dir/                 # Serialized trained model pipelines (.joblib)
├── notebook_dir/               # Exploratory notebooks
├── src/
│   ├── backend/
│   │   ├── api/                # FastAPI route definitions
│   │   ├── config/              # Settings (env-driven)
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── services/            # Prediction logic
│   │   └── main.py              # FastAPI app entrypoint
│   ├── frontend/
│   │   ├── config/              # Frontend settings (API URL)
│   │   ├── pages/                # Streamlit multi-page app
│   │   └── app.py                # Streamlit entrypoint
│   ├── training/                 # Model training scripts + hyperparameter configs
│   └── common/                    # Shared utilities
├── requirements.txt
├── runtime.txt
└── env_template.txt
```

## Running Locally

1. Clone the repo and set up a virtual environment:
   ```bash
   git clone https://github.com/Arpita7007/ML-Multi-Disease-Predictor.git
   cd ML-Multi-Disease-Predictor
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Copy `env_template.txt` to `.env` and fill in the paths (or use absolute local paths).

3. Run the backend:
   ```bash
   uvicorn src.backend.main:app --reload
   ```

4. Run the frontend (in a separate terminal):
   ```bash
   streamlit run src/frontend/app.py
   ```

## Deployment (Render)

Deployed as two independent Render Web Services:

**Backend**
- Build: `pip install -r requirements.txt`
- Start: `uvicorn src.backend.main:app --host 0.0.0.0 --port $PORT`

**Frontend**
- Build: `pip install -r requirements.txt`
- Start: `streamlit run src/frontend/app.py --server.port $PORT --server.address 0.0.0.0`
- Requires `API_URL` env var pointing at the live backend's `/api/predict` endpoint

## Challenges & Fixes

Deploying this project surfaced a few real-world issues beyond just training the models:

- **Module import errors (`Could not import module "src.main"`):** The actual FastAPI entrypoint lived at `src/backend/main.py`, not `src/main.py`. Fixed by correcting the Uvicorn start command to `src.backend.main:app` and ensuring every package directory (`src/`, `src/backend/`) had an `__init__.py`.

- **Corrupted environment variables:** A model path env var got mangled during a dashboard paste, producing a garbled `FileNotFoundError`. Fixed by manually re-entering each path individually instead of bulk-pasting from a `.env` file.

- **`AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`:** The deployed environment installed a newer, incompatible version of scikit-learn than the one used to train and pickle the model (and an unusually new Python 3.14 runtime). Fixed by pinning `scikit-learn==1.7.2` in `requirements.txt` and adding a `runtime.txt` pinning Python to `3.11.9`, matching the training environment.

- **Two-service architecture coordination:** Backend and frontend are deployed as separate Render services. The frontend's `API_URL` had to be updated to the live backend URL post-deployment, since local development defaults to `127.0.0.1`.

## Author

Built by [Arpita7007](https://github.com/Arpita7007) as part of an ML/Data Science portfolio for placement preparation.
