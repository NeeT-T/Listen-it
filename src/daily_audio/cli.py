import os

import click

from daily_audio.output import MarkdownOutput

_QUALITY_HELP = "Qualidade da transcrição: fast (padrão), balanced, best."
_MODEL_HELP = "Modelo Whisper: tiny, base, small, medium, large-v3"


@click.group()
def main():
    """Serviço de captura e transcrição de áudio."""


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
@click.option("--lang", default=None, help="Idioma do áudio (ex: pt, en). Auto-detecta se omitido.")
@click.option("--quality", default="fast", show_default=True, help=_QUALITY_HELP)
@click.option("--no-daemon", is_flag=True, default=False, help="Transcrever sem daemon (recarrega modelo).")
def transcribe(audio_file: str, model: str, lang: str | None, quality: str, no_daemon: bool) -> None:
    """Transcreve um arquivo de áudio existente."""
    click.echo(f"Transcrevendo '{audio_file}' com modelo '{model}' (qualidade: {quality})...")

    if no_daemon:
        from daily_audio.transcriber import transcribe_file
        source = transcribe_file(audio_file, model_name=model, language=lang, quality=quality)
    else:
        from daily_audio.daemon import transcribe_via_daemon
        source = transcribe_via_daemon(audio_file, model_name=model, language=lang, quality=quality)

    with MarkdownOutput(mode="arquivo", model=model, lang=lang or "auto") as out:
        for ts, text in source:
            out.append(ts, text)
            click.echo(f"[{ts}] {text.strip()}")

    click.echo(f"\nSalvo em: {out.path}")


@main.command()
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
@click.option("--lang", default=None, help="Idioma do áudio (ex: pt, en). Auto-detecta se omitido.")
@click.option("--quality", default="fast", show_default=True, help=_QUALITY_HELP)
@click.option("--no-daemon", is_flag=True, default=False, help="Transcrever sem daemon (recarrega modelo).")
def record(model: str, lang: str | None, quality: str, no_daemon: bool) -> None:
    """Grava pelo microfone e transcreve após parar."""
    import threading
    from daily_audio.recorder import record_until_enter

    if no_daemon:
        from daily_audio.transcriber import get_model, transcribe_file
        loader = threading.Thread(target=get_model, args=(model,), daemon=True)
        loader.start()
        audio_path = record_until_enter()
        click.echo(f"\nAguardando modelo '{model}'...")
        loader.join()
        source_fn = lambda p: transcribe_file(p, model_name=model, language=lang, quality=quality)
    else:
        from daily_audio.daemon import ensure_running, transcribe_via_daemon
        # Warm up daemon while user records
        loader = threading.Thread(target=ensure_running, args=(model,), daemon=True)
        loader.start()
        audio_path = record_until_enter()
        click.echo(f"\nAguardando daemon '{model}'...")
        loader.join()
        source_fn = lambda p: transcribe_via_daemon(p, model_name=model, language=lang, quality=quality)

    click.echo(f"Transcrevendo (qualidade: {quality})...")

    with MarkdownOutput(mode="gravação", model=model, lang=lang or "auto") as out:
        for ts, text in source_fn(audio_path):
            out.append(ts, text)
            click.echo(f"[{ts}] {text.strip()}")

    click.echo(f"\nSalvo em: {out.path}")
    os.unlink(audio_path)


@main.command()
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
@click.option("--lang", default="pt", show_default=True, help="Idioma esperado da fala.")
def realtime(model: str, lang: str) -> None:
    """Transcrição em tempo real enquanto você fala."""
    from daily_audio.realtime import run_realtime
    run_realtime(model=model, language=lang)


@main.command("ollama-realtime")
@click.option("--model", default="whisper", show_default=True, help="Modelo Ollama de transcrição (ex: whisper).")
@click.option("--lang", default="pt", show_default=True, help="Idioma esperado da fala.")
@click.option("--url", default="http://localhost:11434", show_default=True, help="URL base do Ollama.")
@click.option("--chunk", default=5.0, show_default=True, type=float, help="Duração máxima do chunk de áudio (segundos).")
def ollama_realtime(model: str, lang: str, url: str, chunk: float) -> None:
    """Transcrição em tempo real via Ollama (envia áudio ao modelo local)."""
    from daily_audio.ollama_realtime import run_ollama_realtime
    run_ollama_realtime(model=model, language=lang, ollama_url=url, chunk_duration=chunk)


@main.command("vosk-realtime")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Caminho para o diretório do modelo Vosk (ex: ~/vosk-model-pt).",
)
@click.option("--lang", default="pt", show_default=True, help="Idioma esperado da fala.")
def vosk_realtime(model_path: str, lang: str) -> None:
    """Transcrição em tempo real offline via Vosk."""
    from daily_audio.vosk_realtime import run_vosk_realtime
    run_vosk_realtime(model_path=model_path, language=lang)


@main.command()
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
def daemon(model: str) -> None:
    """Inicia o daemon de transcrição (mantém o modelo em memória)."""
    from daily_audio.daemon import run_server
    click.echo(f"Iniciando daemon com modelo '{model}'...")
    run_server(model)


@main.command("daemon-stop")
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
def daemon_stop(model: str) -> None:
    """Para o daemon de transcrição."""
    from daily_audio.daemon import stop_daemon
    if stop_daemon(model):
        click.echo(f"Daemon '{model}' encerrado.")
    else:
        click.echo(f"Daemon '{model}' não estava rodando.")


@main.command("daemon-status")
@click.option("--model", default="base", show_default=True, help=_MODEL_HELP)
def daemon_status(model: str) -> None:
    """Verifica se o daemon está rodando."""
    from daily_audio.daemon import _is_running
    if _is_running(model):
        click.echo(f"Daemon '{model}': rodando")
    else:
        click.echo(f"Daemon '{model}': parado")
