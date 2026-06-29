from collections.abc import Generator

from faster_whisper import WhisperModel

_MODEL_CACHE: dict[str, WhisperModel] = {}

BEAM_SIZES = {"fast": 1, "balanced": 3, "best": 5}


def get_model(name: str) -> WhisperModel:
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _MODEL_CACHE[name]


def _format_ts(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def transcribe_file(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    quality: str = "fast",
) -> Generator[tuple[str, str], None, None]:
    model = get_model(model_name)
    beam_size = BEAM_SIZES.get(quality, 1)
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    for seg in segments:
        yield (_format_ts(seg.start), seg.text)
