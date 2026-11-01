# The Maia-3 move service. Build it from THIS directory: it shares nothing with anything around it.
#
#   docker build -t brilliant-chess-maia .
#
# The tests are a stage of their own, run against the real checkpoint, and never part of the image that
# ships (add --no-cache-filter test to run them again on an unchanged tree):
#
#   docker build --target test .
#
# Every RUN stays on one line: this file is checked out with CRLF on Windows, and a continued line would
# hand the shell a stray carriage return.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# The one place that names the model: app.py reads this and takes the architecture from maia3's registry.
ENV MAIA_MODEL=maia3-5m
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The checkpoint, at a pinned Hugging Face revision and checked against a pinned SHA-256, so the running
# service never reaches the network: it starts with none at all.
RUN mkdir /models && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/UofTCSSLab/Maia3-5M/resolve/b6559de2398d7140b985f28fd2c19fb5e47ddabe/maia3-5m.pt', '/models/maia3-5m.pt')"
RUN echo "ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f  /models/maia3-5m.pt" | sha256sum -c -

FROM base AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY app.py .
COPY tests/ tests/
RUN python -m pytest -q -p no:cacheprovider tests

FROM base AS runtime
COPY app.py LICENSE.maia3 ./
EXPOSE 8082
# Torch threads for every forward pass.
ENV MAIA_THREADS=1
# One worker: the model lives in one process and one thread runs every forward pass. The keep-alive
# outlasts Caddy's upstream one (2 minutes by default), so a proxy in front never reuses a connection
# uvicorn has just closed, which would reach the caller as a failed request.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1", "--no-access-log", "--timeout-keep-alive", "130"]
