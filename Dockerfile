FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg git curl patchelf \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir --upgrade yt-dlp

# Copy and run patching script
COPY patch_libs.py .
RUN python3 patch_libs.py

# Check that imports work (optional, but good for debugging)
RUN python3 -c "import ctranslate2; from faster_whisper import WhisperModel; print('imports ok')"
RUN yt-dlp --version

COPY worker.py .

# Same image is used for both GPU (whisper-gpu job def, WHISPER_DEVICE=cuda) and
# CPU (whisper-cpu job def, WHISPER_DEVICE=cpu) Batch job definitions.
CMD ["python3", "worker.py"]
