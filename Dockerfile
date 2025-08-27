FROM ghcr.io/open-webui/pipelines:main-cuda

USER root

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
	&& rm -rf /var/lib/apt/lists/*
