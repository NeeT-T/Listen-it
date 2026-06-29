import json
import queue
from datetime import datetime

import click
import sounddevice as sd

from daily_audio.output import MarkdownOutput

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # ~250ms chunks at 16kHz


def run_vosk_realtime(model_path: str, language: str) -> None:
    import vosk

    vosk.SetLogLevel(-1)

    try:
        model = vosk.Model(model_path)
    except Exception as e:
        raise click.ClickException(f"Falha ao carregar modelo Vosk em '{model_path}': {e}")

    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    out = MarkdownOutput(mode="vosk-tempo-real", model=model_path, lang=language)
    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            click.echo(f"[aviso sounddevice] {status}", err=True)
        audio_queue.put(bytes(indata))

    click.echo(f"Transcrição Vosk em tempo real iniciada. Arquivo: {out.path}")
    click.echo("Fale normalmente. Pressione Ctrl+C para parar.\n")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                data = audio_queue.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"\r[{ts}] {text:<80}")
                        out.append(ts, text)
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if partial:
                        print(f"\r... {partial:<80}", end="", flush=True)
    except KeyboardInterrupt:
        # Flush any remaining audio
        final = json.loads(rec.FinalResult()).get("text", "").strip()
        if final:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\r[{ts}] {final:<80}")
            out.append(ts, final)
        click.echo("\nEncerrando...")
    finally:
        out.close()
        click.echo(f"Salvo em: {out.path}")
