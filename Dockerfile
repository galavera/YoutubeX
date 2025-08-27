FROM ghcr.io/open-webui/pipelines:main

USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

USER 1000
