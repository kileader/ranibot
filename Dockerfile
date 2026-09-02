FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy only application code; credentials are supplied at runtime.
COPY bot.py agents.py llm.py config.py memory.py scenarios.py ./

USER 10001:10001
CMD ["python", "bot.py"]
