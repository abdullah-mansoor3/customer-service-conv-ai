FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend-requirements.txt
COPY rag/requirements.txt ./rag-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r backend-requirements.txt -r rag-requirements.txt

COPY backend/app/ ./app/
COPY rag/ ./rag/
COPY rag_data/ ./rag_data/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
