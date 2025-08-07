from faster_whisper import WhisperModel
import streamlit as st

model = WhisperModel("small", compute_type="int8")


def transcribe_audio(fileName):
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", compute_type="int8")

        segments, info = model.transcribe(fileName)
        final_text = ""
        for segment in segments:
            final_text += segment.text

        return final_text.strip()
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None