# Operations

## Local Services

```bash
uvicorn worldcup_predictor.api.main:app --reload
streamlit run app/dashboard.py
celery -A worldcup_predictor.tasks.celery_app worker --loglevel=info
```

## Docker Compose

```bash
docker compose up --build
```

Services:

- API: <http://localhost:8000>
- Streamlit: <http://localhost:8501>
- PostgreSQL: `postgresql+psycopg://worldcup:worldcup@localhost:5432/worldcup`
- Redis: `redis://localhost:6379/0`

## Data Refresh

```bash
worldcup-predictor fetch-data --output data/raw/international_results.csv
worldcup-predictor train --matches data/raw/international_results.csv --output models/elo_poisson_current.json
worldcup-predictor simulate --model models/elo_poisson_current.json --simulations 1000 --output data/processed/simulation_2026.csv
```

