FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY data/worldcup ./data/worldcup

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "worldcup_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

