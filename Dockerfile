# One image for Railway: build the front end, then serve it and the API from the Python backend.
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app/backend
COPY app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/backend/ ./
COPY --from=frontend /app/frontend/dist /app/frontend/dist
# exec so uvicorn is PID 1 and shuts down cleanly
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
