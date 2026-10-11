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
      pip install "torch==2.11.*" "torchaudio==2.11.*" --index-url "$TORCH_INDEX" \
   && pip install -r requirements-demucs.txt \
   && python -c "from demucs.pretrained import get_model; get_model('htdemucs')"; \
    fi
# Rubber Band's CLI takes a beat map, so every song can be warped onto one exact grid.
RUN apt-get update \
 && apt-get install -y --no-install-recommends rubberband-cli \
 && rm -rf /var/lib/apt/lists/*
# beat_this: neural beat and downbeat tracking, weights baked in like Demucs. It needs torchaudio,
# whose last release is 2.11, hence the torch pin above.
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install "beat-this>=1.1" \
   && python -c "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu')"; \
    fi
# Lyrics: faster-whisper transcribes the separated vocals with word timings, and the CMU pronouncing
# dictionary finds rhymes, so songs can hand over on a shared word or a rhyme. Model baked in.
ARG WHISPER_MODEL=small.en
ENV RAVE_WHISPER_MODEL=$WHISPER_MODEL
RUN pip install "faster-whisper>=1.1" "pronouncing>=0.2" \
 && python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['RAVE_WHISPER_MODEL'], device='cpu', compute_type='int8')"
# Stem packs: 7-Zip unpacks zip/7z/rar/tar uploads. BS-RoFormer (via audio-separator) is the optional
# best-quality vocal separator; torch stays pinned at 2.11 for beat_this, and the model is baked in.
ENV RAVE_ROFORMER_DIR=/opt/models/audio-separator
RUN apt-get update \
 && apt-get install -y --no-install-recommends 7zip \
 && rm -rf /var/lib/apt/lists/*
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      printf 'torch==2.11.*\ntorchaudio==2.11.*\n' > /tmp/torch-pin.txt \
   && apt-get update && apt-get install -y --no-install-recommends gcc g++ \
   && pip install -c /tmp/torch-pin.txt "audio-separator[cpu]" audioread \
   && apt-get purge -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
   && python -c "from audio_separator.separator import Separator; s = Separator(model_file_dir='/opt/models/audio-separator'); s.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')"; \
    fi
# The "best" GPU separation: MelBand-RoFormer vocals + the fine-tuned htdemucs_ft, baked in too.
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      python -c "from audio_separator.separator import Separator; s = Separator(model_file_dir='/opt/models/audio-separator'); s.load_model(model_filename='vocals_mel_band_roformer.ckpt')" \
   && python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"; \
    fi
# Every model is baked in above, so nothing needs the Hugging Face hub at runtime (it would only try to
# rewrite its cache files, which the rave user can't).
ENV HF_HUB_OFFLINE=1

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
