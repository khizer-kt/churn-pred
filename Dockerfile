# Churn Analyst Agent -- Streamlit app, model and agent in one image.
#
# Build:  docker build -t churn-agent .
# Run:    docker run --rm -p 8501:8501 --env-file .env churn-agent
#
# The Groq API key is injected at runtime and never baked into the image.
# Without one the Explore and Score-a-customer tabs still work; only Chat
# needs the language model.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies before source: editing a Python file then rebuilds in seconds
# instead of re-running the install, which is by far the slowest layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pytest.ini ./
COPY src/ ./src/
COPY ui/ ./ui/
COPY tests/ ./tests/
COPY data/ ./data/

# Train during the build rather than COPYing artifacts/ in.
#
# artifacts/ is gitignored, so a fresh clone of this repo does not have it and a
# COPY would fail -- the image has to be buildable by anyone who clones the
# repository. Training takes about 30 seconds, and random_state is fixed
# throughout, so the resulting model is identical on every build.
RUN python -m src.model.train

# Drop root. Streamlit needs a writable home for its config directory.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8501

# python rather than curl: the slim image has no curl, and installing one just
# for a healthcheck is not worth the extra layer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
