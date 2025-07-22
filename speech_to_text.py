from faster_whisper import WhisperModel


model = WhisperModel("small", compute_type="int8")


def transcribe_audio(fileName):
    segments, info = model.transcribe("tmp.m4a")
    finalText = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        finalText += segment.text

    return finalText


def main():
    text = transcribe_audio("tmp.m4a")
    # print(text)


if __name__ == "__main__":
    main()