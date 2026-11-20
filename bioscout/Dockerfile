FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Vector store and SQLite are built at image-build time from data/ (small demo dataset).
# Needs Gemini access during build; for bigger data, build once and store in Cloud Storage instead.
ARG GOOGLE_API_KEY
RUN GOOGLE_API_KEY=$GOOGLE_API_KEY python -m bioscout.ingest
ENV PORT=8080
CMD exec uvicorn bioscout.api:app --host 0.0.0.0 --port ${PORT}
