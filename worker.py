#!/usr/bin/env python3
"""
AWS Batch GPU/CPU worker (one video per job):

Input payload (JSON):
  {
    "video_id": "t5pBl8-xBd8",
    "url": "https://www.youtube.com/watch?v=t5pBl8-xBd8",
    "title": "House Session - 2026-02-27 - 9:30AM",
    "published_at": "2026-02-27T16:10:22Z",
    "Channel_id": "UCC1w34Iyg1vB_HT6dt_4eMA",
    "Municipality": "Vermont House of Representatives",
    "advertising": "<b>...",
    "office_id": "1672",
    "s3_bucket": "statehouses-audio-bucket",
    "s3_key": "jobs/t5pBl8-xBd8.wav",
    "wav_size_mb": 173.62,
    "downloaded_at": "2026-03-03T19:00:39Z"
  }

Required fields: url, title, office_id, s3_bucket, s3_key

The worker will:
- Download pre-converted WAV from S3 (s3_bucket/s3_key)
- Transcribe (faster-whisper on GPU, word timestamps)
- Split into "turn-like" paragraphs based on pauses between segments (no diarization)
- Format HTML transcript with confidence coloring
- Summarize via OpenAI
- POST to Legislata posts API

Primary input method: JOB_PAYLOAD environment variable (set by the Lambda orchestrator
via the Batch job definition container overrides). For local testing, argv[1] or stdin
can be used as fallbacks.

Required env vars:
  OPENAI_API_KEY
  LEGISLATA_API_AUTH_KEY

Optional env vars:
  JOB_PAYLOAD                   JSON job payload (primary input; set by Lambda/Batch)
  POSTS_URL                     default: https://legislata-backend-production.herokuapp.com/public/api/v1/posts
  OPENAI_MODEL                  default: gpt-4o-mini

  WHISPER_MODEL                 default: small
  WHISPER_DEVICE                default: cuda  (falls back through: cuda/float16 → cuda/int8 → cpu/int8_float16 → cpu/int8)
  WHISPER_COMPUTE_TYPE          default: float16
  WHISPER_BEAM_SIZE             default: 1

  TURN_GAP_SECONDS              default: 1.2   (new paragraph if silence gap exceeds this)
  VAD_FILTER                    default: true
  WORD_TIMESTAMPS               default: true

  SUMMARY_CHUNK_CHARS           default: 4000
  SUMMARY_SECTION_TOKENS        default: 200
  MAX_SUMMARY_TOKENS            default: 500

  AWS_REGION / AWS_DEFAULT_REGION  default: us-east-2
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
import tempfile
from typing import Any, Dict, List, Tuple

import boto3
import requests
from openai import OpenAI
from faster_whisper import WhisperModel


# ----------------------------
# Custom exceptions
# ----------------------------

class NonRetryableError(Exception):
    """Raised when a failure is permanent and should not be retried (job will be marked FAILED by Batch)."""


# ----------------------------
# Utilities
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def jitter_sleep(base: float, jitter: float = 0.25) -> None:
    time.sleep(base + random.random() * jitter)

def retry(fn, tries: int = 3, base_delay: float = 2.0, max_delay: float = 15.0, retry_on: Tuple[type, ...] = (Exception,)):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except retry_on as e:
            last = e
            if attempt == tries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            log(f"[retry] {attempt}/{tries} failed: {e} (sleep {delay:.1f}s)")
            jitter_sleep(delay)
    raise last

def read_job_payload() -> Dict[str, Any]:
    """Read job payload from argv, stdin, or JOB_PAYLOAD environment variable.

    Primary path (Batch jobs): JOB_PAYLOAD env var set by the Lambda orchestrator
    via container overrides in the Batch job definition.

    Fallback paths (local testing): argv[1] or stdin.
    """
    raw = None

    # Try command-line argument first
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        raw = sys.argv[1]
        log("[INPUT] Payload from argv[1]")
    # Try stdin
    elif not sys.stdin.isatty():
        stdin_raw = sys.stdin.read()
        if stdin_raw and stdin_raw.strip():
            raw = stdin_raw
            log("[INPUT] Payload from stdin")

    # Primary path: JOB_PAYLOAD env var (set by Lambda/Batch)
    if not raw or not raw.strip():
        env_payload = os.environ.get("JOB_PAYLOAD", "").strip()
        if env_payload:
            raw = env_payload
            log("[INPUT] Payload from JOB_PAYLOAD env var")

    if not raw or not raw.strip():
        raise ValueError("No job payload found in argv[1], stdin, or JOB_PAYLOAD env var")

    return json.loads(raw)


# ----------------------------
# S3 download
# ----------------------------

def download_wav_from_s3(s3_bucket: str, s3_key: str, out_wav_path: str) -> None:
    """Download a pre-converted WAV file from S3."""
    region = os.environ.get('AWS_REGION', os.environ.get('AWS_DEFAULT_REGION', 'us-east-2'))
    log(f"[s3] Downloading s3://{s3_bucket}/{s3_key} -> {out_wav_path}")
    s3 = boto3.client('s3', region_name=region)
    def _download():
        s3.download_file(s3_bucket, s3_key, out_wav_path)
    retry(_download, tries=3, base_delay=2.0)
    size_mb = os.path.getsize(out_wav_path) / (1024 * 1024)
    log(f"[s3] Downloaded {size_mb:.1f} MB")


def delete_s3_file(s3_bucket: str, s3_key: str) -> None:
    """Delete the audio file from S3 after successful processing."""
    region = os.environ.get('AWS_REGION', os.environ.get('AWS_DEFAULT_REGION', 'us-east-2'))
    log(f"[s3] Deleting s3://{s3_bucket}/{s3_key}")
    s3 = boto3.client('s3', region_name=region)
    try:
        s3.delete_object(Bucket=s3_bucket, Key=s3_key)
        log(f"[s3] Deleted s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        log(f"[s3] Warning: Failed to delete s3://{s3_bucket}/{s3_key}: {e}")


# ----------------------------
# Transcription (no diarization)
# ----------------------------

def transcribe_words_and_segments(wav_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      words: [{start,end,word,confidence}]
      segs:  [{start,end,text}]  (used for pause-based paragraphing)
    """
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))

    vad_filter = os.environ.get("VAD_FILTER", "true").strip().lower() in ("1","true","t","yes","y")
    word_ts = os.environ.get("WORD_TIMESTAMPS", "true").strip().lower() in ("1","true","t","yes","y")

    FALLBACK_CHAIN = [
        ("cuda",  "float16"),
        ("cuda",  "int8"),
        ("cpu",   "int8_float16"),
        ("cpu",   "int8"),
    ]

    start_device = os.environ.get("WHISPER_DEVICE", "cuda")
    start_compute = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

    try:
        start_idx = FALLBACK_CHAIN.index((start_device, start_compute))
    except ValueError:
        start_idx = 0

    attempts = FALLBACK_CHAIN[start_idx:]
    if ("cpu", "int8") not in attempts:
        attempts = attempts + [("cpu", "int8")]

    last_exc: Exception = RuntimeError("Transcription failed before any attempts could be made")
    for device, compute_type in attempts:
        log(f"[whisper] attempting device={device} compute_type={compute_type} model={whisper_model}")
        try:
            model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        except Exception as e:
            log(f"[whisper] device={device} compute_type={compute_type} failed: {e}, trying next option...")
            last_exc = e
            continue

        log(f"[whisper] model loaded successfully on device={device} compute_type={compute_type}")

        # Determine the VAD filter values to try for this device/compute_type.
        # If VAD is enabled, attempt with VAD first; if that fails or returns no
        # segments, retry without VAD before falling through to the next device.
        vad_filter_values = [True, False] if vad_filter else [False]

        for use_vad in vad_filter_values:
            log(f"[whisper] transcribing… beam_size={beam_size} vad_filter={use_vad} word_timestamps={word_ts}")
            try:
                segments, _info = model.transcribe(
                    wav_path,
                    vad_filter=use_vad,
                    word_timestamps=word_ts,
                    beam_size=beam_size,
                )

                words: List[Dict[str, Any]] = []
                segs: List[Dict[str, Any]] = []

                seg_count = 0
                for seg in segments:
                    seg_count += 1
                    segs.append({"start": float(seg.start), "end": float(seg.end), "text": (seg.text or "").strip()})

                    if word_ts and seg.words:
                        for w in seg.words:
                            prob = float(w.probability) if w.probability is not None else 0.0
                            words.append({
                                "start": float(w.start),
                                "end": float(w.end),
                                "word": w.word,
                                "confidence": prob,
                            })

                if seg_count == 0 and use_vad:
                    log(f"[whisper] vad_filter=True returned 0 segments; retrying with vad_filter=False…")
                    continue

                log(f"[whisper] segments={seg_count} words={len(words)} vad_filter={use_vad}")
                return words, segs
            except Exception as e:
                if use_vad and "empty sequence" in str(e):
                    log(f"[whisper] vad_filter=True raised '{e}'; retrying with vad_filter=False…")
                    last_exc = e
                    continue
                log(f"[whisper] device={device} compute_type={compute_type} failed: {e}, trying next option...")
                last_exc = e
                break

    raise RuntimeError(f"All transcription attempts failed. Tried: {attempts}. Last error: {last_exc}")


# ----------------------------
# Formatting helpers
# ----------------------------

def fmt_timestamp(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(round(s % 60))
    return f"{h}h{m}m{sec}s"

def color_word(word: str, conf: float) -> str:
    if conf > 0.9:
        return word
    if conf > 0.5:
        return f"<span style='color:blueviolet'>{word}</span>"
    return f"<span style='color:red'>{word}</span>"

def build_paragraphs_from_words(words: List[Dict[str, Any]], turn_gap: float) -> List[Dict[str, Any]]:
    """
    Create "turn-like" paragraphs by splitting when the silence gap between words exceeds turn_gap.
    Returns list of paragraphs: [{start, html_text}]
    """
    if not words:
        return []

    words = sorted(words, key=lambda x: x["start"])
    paras: List[Dict[str, Any]] = []

    cur_words: List[str] = []
    cur_start = float(words[0]["start"])
    prev_end = float(words[0]["end"])

    def flush():
        nonlocal cur_words, cur_start
        if cur_words:
            paras.append({"start": cur_start, "html": " ".join(cur_words).strip()})
        cur_words = []

    for w in words:
        gap = float(w["start"]) - float(prev_end)
        if cur_words and gap > turn_gap:
            flush()
            cur_start = float(w["start"])
        cur_words.append(color_word(w["word"], float(w.get("confidence", 0.0))))
        prev_end = float(w["end"])

    flush()
    return paras

def build_transcript_html(words: List[Dict[str, Any]]) -> str:
    if not words:
        return "<p><i>No transcript produced.</i></p>"

    turn_gap = float(os.environ.get("TURN_GAP_SECONDS", "1.2"))
    paras = build_paragraphs_from_words(words, turn_gap=turn_gap)

    parts: List[str] = []
    for p in paras:
        ts = fmt_timestamp(p["start"])
        parts.append(f"<br><br><b>[{ts}]</b> {p['html']}")
    return "".join(parts)

def embed_html_from_url(url: str) -> str:
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    vid = m.group(1) if m else ""
    if not vid:
        return ""
    return (
        '<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 20px;">'
        '<iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" '
        f'src="https://www.youtube.com/embed/{vid}" '
        'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        "</div>"
    )

def strip_html(text: str) -> str:
    return re.sub(r"<.*?>", "", text)

def chunk_text(text: str, max_chars: int) -> List[str]:
    t = strip_html(text)
    paras = re.split(r"\n\s*\n", t)
    chunks: List[str] = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            sents = re.split(r"(?<=[\.\?\!])\s+", c)
            cur2 = ""
            for s in sents:
                if len(cur2) + len(s) + 1 <= max_chars:
                    cur2 = (cur2 + " " + s).strip()
                else:
                    if cur2:
                        final.append(cur2)
                    cur2 = s
            if cur2:
                final.append(cur2)
    return final

def summarize_transcript(openai_client: OpenAI, transcript_html: str) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    max_final = int(os.environ.get("MAX_SUMMARY_TOKENS", "500"))
    section_tokens = int(os.environ.get("SUMMARY_SECTION_TOKENS", "200"))
    chunk_chars = int(os.environ.get("SUMMARY_CHUNK_CHARS", "4000"))

    chunks = chunk_text(transcript_html, max_chars=chunk_chars)
    log(f"[sum] chunks={len(chunks)}")

    def chat(prompt: str, content: str, max_tokens: int) -> str:
        r = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You summarize municipal/government meeting transcripts for policy professionals."},
                {"role": "user", "content": f"{prompt}\n\n{content}"},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()

    if len(chunks) == 1:
        return chat("Create a concise summary of this transcript:", chunks[0], max_final)

    summaries: List[str] = []
    for i, c in enumerate(chunks, start=1):
        log(f"[sum] section {i}/{len(chunks)}")
        summaries.append(chat("Create a concise summary of this section of a transcript:", c, section_tokens))
        jitter_sleep(0.4)

    return chat("Create a concise overall summary from these section summaries:", "\n\n".join(summaries), max_final)

def build_full_description(summary: str, advertising_html: str, embed_html: str, transcript_html: str) -> str:
    disclosure = " <i>This summary was created by AI and may contain hallucinations.</i><br><br>"
    explainer = (
        "<p>This is an AI generated transcript from the original video. "
        "Words are color coded by confidence of the AI model. Words in black have a greater than 90% confidence, "
        "between 50% and 90% in <span style='color:blueviolet'> purple </span> and less than 50% in <span style='color:red'>red</span>.</p>"
    )
    adv = (advertising_html or "").strip()
    adv_block = (adv + "<br><br>") if adv else ""
    return summary + disclosure + adv_block + (embed_html or "") + explainer + transcript_html


# ----------------------------
# POST to Legislata API
# ----------------------------

def post_to_api(payload: Dict[str, Any]) -> None:
    url = os.environ.get("POSTS_URL", "https://legislata-backend-production.herokuapp.com/public/api/v1/posts")
    key = os.environ.get("LEGISLATA_API_AUTH_KEY")
    if not key:
        raise RuntimeError("LEGISLATA_API_AUTH_KEY env var is required")

    headers = {"Authorization": f"API_AUTH_KEY {key}", "Content-Type": "application/json"}

    def _do():
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if r.status_code >= 300:
            raise RuntimeError(f"POST failed {r.status_code}: {r.text[:800]}")
        return r

    retry(_do, tries=3, base_delay=2.0)


# ----------------------------
# Main
# ----------------------------

def process_one(job: Dict[str, Any]) -> None:
    url = job.get("url")
    if not url:
        raise ValueError("job payload must include 'url'")

    title = job.get("title")
    if not title:
        raise ValueError("job payload must include 'title'")

    office_id_raw = job.get("office_id")
    if office_id_raw is None:
        raise ValueError("job payload must include 'office_id'")
    office_id = int(office_id_raw)

    s3_bucket = job.get("s3_bucket")
    if not s3_bucket:
        raise ValueError("job payload must include 's3_bucket'")

    s3_key = job.get("s3_key")
    if not s3_key:
        raise ValueError("job payload must include 's3_key'")

    video_id = job.get("video_id", "")
    municipality = job.get("Municipality", "").strip()
    advertising = job.get("advertising", "")
    public = True

    log(f"[meta] video_id={video_id!r} office_id={office_id} municipality={municipality!r} title={title[:120]!r}")
    log(f"[meta] s3_bucket={s3_bucket!r} s3_key={s3_key!r}")

    user_id = 2
    final_title = f"{municipality}: {title}".strip(": ").strip() if municipality else title

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY env var is required")
    oa = OpenAI(api_key=openai_key)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")

        t0 = time.time()
        download_wav_from_s3(s3_bucket, s3_key, wav_path)
        log(f"[time] download+wav {time.time()-t0:.1f}s")

        t1 = time.time()
        words, _segs = transcribe_words_and_segments(wav_path)
        log(f"[time] transcribe {time.time()-t1:.1f}s")

        transcript_html = build_transcript_html(words)
        embed = embed_html_from_url(url)

        t2 = time.time()
        summary = summarize_transcript(oa, transcript_html)
        log(f"[time] summarize {time.time()-t2:.1f}s")

        description = build_full_description(summary, advertising, embed, transcript_html)

        api_payload = {
            "office_id": office_id,
            "user_id": user_id,
            "title": final_title,
            "description": description,
            "public": public,
        }

        post_to_api(api_payload)
        log("[POST] OK")

        # Clean up the S3 audio file now that it has been successfully posted
        delete_s3_file(s3_bucket, s3_key)


def main() -> None:
    job = read_job_payload()
    log(f"[job] keys={list(job.keys())}")

    jitter_sleep(0.5)

    try:
        process_one(job)
        log("[DONE]")
    except NonRetryableError as e:
        log(f"[ERROR] Non-retryable failure: {e}")
        raise
    except Exception as e:
        log(f"[ERROR] Job processing failed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        raise
