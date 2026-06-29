import tempfile
import threading

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000


def record_until_enter() -> str:
    """Record from microphone until user presses Enter. Returns path to temp wav file."""
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _callback(indata, frames, time, status):
        if not stop_event.is_set():
            chunks.append(indata.copy())

    print("Gravando... Pressione Enter para parar.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_callback):
        input()
        stop_event.set()

    audio = np.concatenate(chunks, axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name
