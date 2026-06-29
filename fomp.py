from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print("Você disse:", text)

if __name__ == "__main__":
    recorder = AudioToTextRecorder()
    print("Diga algo...")
    while True:
        recorder.text(process_text)
