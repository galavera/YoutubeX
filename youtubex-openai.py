"""
title: YoutubeX
author: Galavera
date: 2025-08-27
version: 1.2
license: MIT
description: A custom pipeline that performs YouTube video search, transcribes audio with WhisperX (small.en by default),
             generates transcript summaries, conducts Q&A over transcripts, and searches within transcript/video content.
requirements: torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1, yt-dlp, imageio-ffmpeg, whisperx, pydantic==2.7.4, requests, langchain==0.3.3, langchain-community==0.3.2, langchain-openai==0.2.2, langchain-core==0.3.10, langchain-text-splitters==0.3.0, httpx==0.27.*
"""

import os
import json
import requests
import tempfile
import subprocess
from typing import List, Sequence

from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
import yt_dlp
import imageio_ffmpeg
import whisperx

# Reminder to RUN pip install --upgrade pip && pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# =========================
# YouTube Search Tool
# =========================


class YoutubeSearchInput(BaseModel):
    query: str = Field(description="Search query for YouTube videos.")


@tool("search_youtube", args_schema=YoutubeSearchInput, return_direct=False)
def search_youtube(query: str) -> str:
    """
    Search YouTube for videos and return a JSON array of video URLs.
    Uses yt-dlp's built-in search (ytsearch).
    """
    try:
        ydl_opts = {
            "quiet": True,
            "nocheckcertificate": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch10:{query}", download=False)
        urls = [e["webpage_url"] for e in info.get("entries", []) if "webpage_url" in e]
        return json.dumps(urls)
    except Exception as e:
        return f"Search failed: {e}"


# =========================
# WhisperX Transcription Tool (replaces get_youtube_transcript)
# - Uses yt-dlp Python API to resolve a direct audio stream URL
# - Uses imageio-ffmpeg’s bundled ffmpeg to convert to WAV
# - Transcribes with WhisperX (no SciPy alignment; fewer dependencies)
# =========================


class WhisperXInput(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # silence "model_" warnings in Pydantic
    youtube_url: str = Field(description="YouTube video URL")
    # Defaults suitable for English summaries; override via tool args if needed
    size: str = Field(
        default=os.getenv("WHISPERX_MODEL", "small.en"),
        description="WhisperX model (e.g., small.en)",
    )
    language: str | None = Field(
        default=os.getenv("WHISPERX_LANG", "en"),
        description="Language hint, e.g., 'en'",
    )
    compute_type: str = Field(
        default=os.getenv("WHISPERX_COMPUTE", "auto"),
        description="auto|float16|int8|float32",
    )


def _pick_bestaudio(info: dict) -> str | None:
    """Choose a good audio-only stream (m4a/webm) by bitrate; fallback to top-level URL."""
    fmts = info.get("formats") or []
    audio_only = [
        f
        for f in fmts
        if f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")
    ]
    audio_only.sort(key=lambda f: (f.get("abr") or 0), reverse=True)
    if audio_only:
        return audio_only[0].get("url")
    return info.get("url")


@tool("transcribe_whisperx", args_schema=WhisperXInput, return_direct=False)
def transcribe_whisperx(
    youtube_url: str,
    size: str,
    language: str | None,
    compute_type: str = "auto",
) -> str:
    """Download audio via yt-dlp, convert to 16k mono WAV with imageio-ffmpeg, and transcribe with WhisperX."""
    import shutil, traceback, textwrap

    try:
        tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

        # 1) Download bestaudio without yt-dlp postprocessors (no ffprobe needed)
        ydl_opts = {
            "quiet": True,
            "nocheckcertificate": True,
            "noplaylist": True,
            "skip_download": False,
            "outtmpl": outtmpl,
            "format": "bestaudio[acodec=opus]/bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best",
            "concurrent_fragment_downloads": 3,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            in_path = ydl.prepare_filename(info)  # e.g., .../VIDEOID.webm or .m4a

        if not os.path.exists(in_path):
            candidates = [
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.lower().endswith((".webm", ".m4a", ".mp4", ".opus", ".mp3"))
            ]
            in_path = max(candidates, key=os.path.getsize) if candidates else None
        if not in_path or not os.path.exists(in_path):
            return "Audio download/convert failed: input file not found."

        # 2) Convert to 16k mono WAV using imageio-ffmpeg's ffmpeg
        wav_path = os.path.splitext(in_path)[0] + ".wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            in_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            wav_path,
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", "ignore")
            return "Transcription failed: ffmpeg conversion error: " + textwrap.shorten(
                err, width=1200
            )

        if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024):
            return "Conversion to WAV failed: output too small or missing."

        # 3) Decide device & compute type; require cuDNN for CUDA to avoid half-initialized crashes
        import torch as _t

        try:
            use_cuda = bool(_t.cuda.is_available())
        except Exception:
            use_cuda = False
        try:
            use_cudnn = bool(_t.backends.cudnn.is_available())
        except Exception:
            use_cudnn = False

        device = "cuda" if (use_cuda and use_cudnn) else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # 4) Load model with graceful CUDA→CPU fallback on cuDNN/graph errors
        try:
            model = whisperx.load_model(
                size, device, compute_type=compute_type, language=language
            )
        except OSError as e:
            emsg = str(e).lower()
            if any(
                k in emsg
                for k in (
                    "cudnn",
                    "ops_infer",
                    "cudnn_graph",
                    "invalid handle",
                    "libcudnn",
                )
            ):
                device, compute_type = "cpu", "int8"
                model = whisperx.load_model(
                    size, device, compute_type=compute_type, language=language
                )
            else:
                raise

        # 5) Transcribe
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=8)

        segs = result.get("segments") or []
        text = " ".join((s.get("text") or "").strip() for s in segs if s.get("text"))
        return text.strip() or "(empty transcription)"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Transcription failed: {e}\n{tb}"
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# =========================
# Main Pipeline
# =========================


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        OPENAI_API_MODEL: str = "gpt-5-mini"
        OPENAI_API_TEMPERATURE: float = 1
        AGENT_SYSTEM_PROMPT: str = (
            "You are an intelligent assistant that helps users interact with YouTube videos. "
            "Your tasks include searching for videos, retrieving transcripts, and transforming transcripts "
            "into clear, structured insights. When summarizing, remove filler and repetition, highlight key "
            "points, and present information concisely in the style requested. Stay grounded in the transcript."
        )

    def __init__(self):
        self.name = "YouTubeX"
        self.tools = None
        self.valves = self.Valves(OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""))
        self.pipelines = self.get_openai_models()

    def get_openai_models(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                response = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )
                models = response.json()
                return [
                    {"id": m["id"], "name": m.get("name", m["id"])}
                    for m in models["data"]
                    if "gpt" in m["id"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [{"id": "error", "name": "Could not fetch models from OpenAI."}]
        return []

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        try:
            model = ChatOpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                model=self.valves.OPENAI_API_MODEL,
                temperature=self.valves.OPENAI_API_TEMPERATURE,
            )
            tools: Sequence[BaseTool] = [search_youtube, transcribe_whisperx]
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.valves.AGENT_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("user", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
            res = agent_executor.invoke(
                {"input": user_message, "chat_history": messages}
            )
            return res.get("output", res)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
