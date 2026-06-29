from datetime import datetime

import click

from daily_audio.output import MarkdownOutput


def run_realtime(model: str, language: str | None) -> None:
    from RealtimeSTT import AudioToTextRecorder

    lang = language or "pt"
    out = MarkdownOutput(mode="tempo-real", model=model, lang=lang)

    click.echo(f"Transcrição em tempo real iniciada. Arquivo: {out.path}")
    click.echo("Fale normalmente. Pressione Ctrl+C para parar.\n")

    def on_partial(text: str) -> None:
        print(f"\r... {text.strip():<80}", end="", flush=True)

    def on_final(text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\r[{ts}] {text.strip():<80}")
        out.append(ts, text)

    recorder = AudioToTextRecorder(
        model=model,
        language=lang,
        device="cuda",
        compute_type="float16",
        spinner=False,
        silero_sensitivity=0.4,
        post_speech_silence_duration=0.5,
        # habilita transcrição parcial enquanto a pessoa fala
        enable_realtime_transcription=True,
        use_main_model_for_realtime=True,
        realtime_processing_pause=1,
        on_realtime_transcription_update=on_partial,
    )

    try:
        # loop recomendado pela doc: text(callback) bloqueia até fim do enunciado,
        # dispara on_partial durante a fala, chama on_final ao finalizar
        while True:
            recorder.text(on_final)
    except KeyboardInterrupt:
        click.echo("\nEncerrando...")
        recorder.shutdown()
        out.close()
        click.echo(f"Salvo em: {out.path}")
