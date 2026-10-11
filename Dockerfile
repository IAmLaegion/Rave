# Rave: Pitch Perfect-style mashups, served as a web app on port 8000.
#
#   docker build -t rave .                      # with Demucs (CPU)
#   docker build -t rave --build-arg WITH_DEMUCS=0 .   # small image, fast separator only
#   docker build -t rave:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
FROM denoland/deno:bin-2.5.6 AS deno

FROM python:3.12-slim

ARG WITH_DEMUCS=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Debian's ffmpeg is built with Rubber Band, which does the tempo/pitch matching.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JavaScript runtime to solve YouTube's player challenges.
COPY --from=deno /deno /usr/local/bin/deno

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    TORCH_HOME=/opt/models/torch \
    NUMBA_CACHE_DIR=/tmp/numba \
    RAVE_DATA=/data \
    YTDLP_COOKIES=/data/cookies.txt \
    PATH=/home/rave/.local/bin:$PATH

WORKDIR /app
COPY requirements.txt requirements-demucs.txt ./
RUN pip install -r requirements.txt
# Demucs and its model weights are baked into the image so the first mashup doesn't wait on a download.
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install torch --index-url "$TORCH_INDEX" \
   && pip install -r requirements-demucs.txt \
   && python -c "from demucs.pretrained import get_model; get_model('htdemucs')"; \
    fi

COPY rave ./rave
COPY docker-entrypoint.sh /usr/local/bin/
RUN ffmpeg -hide_banner -filters | grep -q rubberband \
 && useradd --create-home --uid 1000 rave \
 && mkdir -p /data && chown rave /data

USER rave
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/info')"
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "rave.web:app", "--host", "0.0.0.0", "--port", "8000"]
