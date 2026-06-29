import base64
import io
import queue
import threading
from datetime import datetime

import click
import httpx
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from daily_audio.output import MarkdownOutput

SAMPLE_RATE = 16000
_SILENCE_THRESHOLD = 300  # RMS below this is considered silence
_SILENCE_CHUNKS = 8       # ~800ms of consecutive silence triggers send


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def _audio_to_base64_wav(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio.astype(np.int16))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_ollama_realtime(
    model: str,
    language: str,
    ollama_url: str,
    chunk_duration: float,
) -> None:
    out = MarkdownOutput(mode="tempo-real-ollama", model=model, lang=language)
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    running = threading.Event()
    running.set()

    click.echo(f"Transcrição via Ollama iniciada. Arquivo: {out.path}")
    click.echo(f"Modelo: {model} | URL: {ollama_url}")
    click.echo("Fale normalmente. Pressione Ctrl+C para parar.\n")

    def audio_callback(indata, frames, time, status):
        if running.is_set():
            audio_queue.put(indata.copy())

    def transcribe_chunk(audio: np.ndarray) -> None:
        if len(audio) < SAMPLE_RATE // 4:  # skip < 250ms
            return

        # karanchopda333/whisper uses /api/generate with audio as base64 in "images"
        payload = {
            "model": model,
            "prompt": f"Transcribe the audio. Language: {language}.",
            "stream": False,
            "images": [_audio_to_base64_wav(audio)],
        }
        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if text:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{ts}] {text:<80}")
                out.append(ts, text)
        except httpx.HTTPStatusError as e:
            click.echo(f"\nErro HTTP {e.response.status_code}: {e.response.text[:200]}", err=True)
        except httpx.HTTPError as e:
            click.echo(f"\nErro HTTP: {e}", err=True)
        except Exception as e:
            click.echo(f"\nErro: {e}", err=True)

    def processor_loop() -> None:
        buffer: list[np.ndarray] = []
        silent_count = 0
        max_samples = int(SAMPLE_RATE * chunk_duration)

        while True:
            try:
                block = audio_queue.get(timeout=1.0)
            except queue.Empty:
                if not running.is_set() and buffer:
                    transcribe_chunk(np.concatenate(buffer, axis=0).flatten())
                    buffer = []
                if not running.is_set():
                    break
                continue

            if block is None:
                if buffer:
                    transcribe_chunk(np.concatenate(buffer, axis=0).flatten())
                break

            buffer.append(block)
            total_samples = sum(len(b) for b in buffer)

            if _rms(block) < _SILENCE_THRESHOLD:
                silent_count += 1
            else:
                silent_count = 0

            if (silent_count >= _SILENCE_CHUNKS or total_samples >= max_samples) and buffer:
                audio_data = np.concatenate(buffer, axis=0).flatten()
                buffer = []
                silent_count = 0
                threading.Thread(target=transcribe_chunk, args=(audio_data,), daemon=True).start()

    proc = threading.Thread(target=processor_loop, daemon=True)
    proc.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
    ):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            click.echo("\nEncerrando...")
            running.clear()
            audio_queue.put(None)
            proc.join(timeout=10)
            out.close()
            click.echo(f"Salvo em: {out.path}")
