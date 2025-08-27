FROM ghcr.io/open-webui/pipelines:main

USER root

COPY youtubex-openai.py /app/pipelines/
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
	&& rm -rf /var/lib/apt/lists/*
