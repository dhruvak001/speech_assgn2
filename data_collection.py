#!/usr/bin/env python3
"""
data_collection.py
==================
Downloads a 10-minute audio segment from the Speech Understanding lecture
(2:20:00 – 2:30:00, i.e. seconds 8400–9000) without fetching the full video.

Strategy
--------
1. Use yt-dlp  -g  to obtain the direct audio-stream URL (no local download).
2. Pipe that URL through ffmpeg, seeking to the target timestamp, and writing
   a PCM WAV file at the requested sample rate.

Requirements: yt-dlp, ffmpeg (both must be on PATH).
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Default extraction window ─────────────────────────────────────────────────
DEFAULT_URL      = "https://youtu.be/ZPUtA3W-7_I"
DEFAULT_START    = 8400   # 2 h 20 min  →  seconds
DEFAULT_DURATION = 600    # 10 minutes
DEFAULT_OUT_ASR  = "data/original_segment.wav"      # 16 kHz  (ASR / LID)
DEFAULT_OUT_TTS  = "data/original_segment_22k.wav"  # 22.05 kHz (prosody ref)


# ── helpers ─────────────────────────────────────────────────────────────────

def _yt_dlp_cmd() -> list:
    """Return yt-dlp invocation, preferring installed binary then python -m."""
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    # Fall back to python -m yt_dlp (works when installed via pip)
    return [sys.executable, "-m", "yt_dlp"]


def _get_stream_url(youtube_url: str) -> str:
    """Ask yt-dlp for the best-audio direct URL (no local download)."""
    cmd = _yt_dlp_cmd() + [
        "-g",                    # print URL only
        "-f", "bestaudio/best",
        "--no-playlist",
        youtube_url,
    ]
    log.info("Resolving stream URL via yt-dlp …")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    url = result.stdout.strip().split("\n")[0]
    log.info(f"Stream URL obtained (length={len(url)} chars).")
    return url


def _ffmpeg_extract(stream_url: str,
                    start_sec: int,
                    duration_sec: int,
                    output_path: str,
                    sample_rate: int) -> None:
    """Use ffmpeg to seek+extract exactly the desired window."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-ss",  str(start_sec),       # fast-seek BEFORE -i  → very efficient
        "-i",   stream_url,
        "-t",   str(duration_sec),
        "-vn",                         # drop video
        "-ar",  str(sample_rate),
        "-ac",  "1",                   # mono
        "-acodec", "pcm_s16le",
        "-y",
        output_path,
    ]

    log.info(
        f"Extracting {duration_sec}s starting at {start_sec}s "
        f"→ {output_path}  (sr={sample_rate})"
    )
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(output_path) / 1e6
    log.info(f"Saved {output_path}  ({size_mb:.1f} MB).")


# ── public API ───────────────────────────────────────────────────────────────

def download_segment(
    url: str       = DEFAULT_URL,
    start_sec: int = DEFAULT_START,
    duration_sec: int = DEFAULT_DURATION,
    output_asr: str = DEFAULT_OUT_ASR,
    output_tts: str = DEFAULT_OUT_TTS,
) -> dict:
    """
    Download one lecture segment at two sample rates.

    Returns
    -------
    dict with keys "asr" and "tts" pointing to the saved WAV paths.
    """
    stream_url = _get_stream_url(url)

    # ASR / LID quality – 16 kHz mono
    _ffmpeg_extract(stream_url, start_sec, duration_sec, output_asr, 16_000)

    # Prosody reference + TTS quality – 22.05 kHz mono
    _ffmpeg_extract(stream_url, start_sec, duration_sec, output_tts, 22_050)

    return {"asr": output_asr, "tts": output_tts}


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a timed audio segment from a YouTube lecture."
    )
    parser.add_argument("--url",      default=DEFAULT_URL)
    parser.add_argument("--start",    type=int, default=DEFAULT_START,
                        help="Start offset in seconds (default 8400 = 2h20m)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Clip length in seconds (default 600 = 10 min)")
    parser.add_argument("--out-asr",  default=DEFAULT_OUT_ASR)
    parser.add_argument("--out-tts",  default=DEFAULT_OUT_TTS)
    args = parser.parse_args()

    paths = download_segment(
        url          = args.url,
        start_sec    = args.start,
        duration_sec = args.duration,
        output_asr   = args.out_asr,
        output_tts   = args.out_tts,
    )
    print(f"\nDone.\n  ASR file : {paths['asr']}\n  TTS file : {paths['tts']}")
