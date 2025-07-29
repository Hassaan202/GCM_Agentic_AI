import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import pandas as pd
import os
import google.generativeai as genai
import json
from speech_to_text import transcribe_audio
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
# Add your API key here or use environment variable
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize session state
if 'food_logs' not in st.session_state:
    st.session_state.food_logs = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'pending_food_items' not in st.session_state:
    st.session_state.pending_food_items = []


def record_audio(duration=5, sample_rate=44100):
    """Record audio for specified duration"""
    try:
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        audio_data = sd.rec(int(duration * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None, None


def save_audio(audio_data, sample_rate, filename="tmp.m4a"):
    """Save audio data to file"""
    try:
        # Convert to wav first (soundfile doesn't support m4a directly)
        temp_wav = "temp_recording.wav"
        sf.write(temp_wav, audio_data, sample_rate)
        return temp_wav
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None


def process_food_with_gemini(transcribed_text):
    """Use Gemini API to extract food items and calories"""
    if not GEMINI_API_KEY:
        st.error("Gemini API key not configured. Please add your API key.")
        return None

    try:
        prompt = f"""
        Analyze the following text and extract food items with their typical serving sizes and calories per serving.

        Text: "{transcribed_text}"

        Please respond with a JSON array where each food item is an object with these exact keys:
        - "food_name": string (name of the food item)
        - "typical_serving": string (typical serving size, e.g., "1 cup", "1 slice", "100g")
        - "calories_per_serving": number (calories in the typical serving)

        Example format:
        [
            {{
                "food_name": "chicken breast",
                "typical_serving": "100g",
                "calories_per_serving": 165
            }},
            {{
                "food_name": "rice",
                "typical_serving": "1 cup cooked",
                "calories_per_serving": 205
            }}
        ]

        If no food items are found, return an empty array [].
        Only return the JSON array, no other text.
        """

        response = model.generate_content(prompt)

        # Parse the JSON response
        try:
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ending ```

            response_text = response_text.strip()

            food_items = json.loads(response_text)
            return food_items
        except json.JSONDecodeError as e:
            st.error("Error parsing Gemini response. Please try again.")
            st.write("Raw response:", response.text)
            st.write("Cleaned response:", response_text)
            st.write("JSON Error:", str(e))
            return None

    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None


def display_food_confirmation(food_items):
    """Display food items for confirmation and serving quantity input"""
    if not food_items:
        return []

    confirmed_items = []

    st.subheader("🍎 Confirm Food Items")
    st.write("Please confirm the detected food items and enter serving quantities:")

    for i, item in enumerate(food_items):
        with st.expander(f"📝 {item['food_name'].title()}", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**Food:** {item['food_name']}")
                st.write(f"**Typical serving:** {item['typical_serving']}")
                st.write(f"**Calories per serving:** {item['calories_per_serving']}")

            with col2:
                serving_quantity = st.number_input(
                    f"Servings consumed",
                    min_value=0.1,
                    max_value=50.0,
                    value=1.0,
                    step=0.1,
                    key=f"serving_{i}"
                )

            with col3:
                total_calories = serving_quantity * item['calories_per_serving']
                st.metric("Total Calories", f"{total_calories:.0f}")

            # Add confirm button for each item
            if st.button(f"✅ Add {item['food_name']}", key=f"confirm_{i}"):
                confirmed_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'food_name': item['food_name'],
                    'typical_serving': item['typical_serving'],
                    'calories_per_serving': item['calories_per_serving'],
                    'serving_quantity': serving_quantity,
                    'total_calories': total_calories
                }
                st.session_state.food_logs.append(confirmed_item)
                st.success(f"✅ Added {item['food_name']} to your food log!")

    return confirmed_items


def main():
    st.title("🍽️ Voice Food Logger with AI Analysis")
    st.markdown("Record what you eat and get automatic calorie tracking using AI!")

    # API Key configuration
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key not configured. Please add your API key to use AI food analysis.")
        api_key_input = st.text_input("Enter your Gemini API Key:", type="password")
        if api_key_input:
            st.session_state.temp_api_key = api_key_input
            st.info("API key entered. Please restart the app or use secrets.toml for permanent configuration.")

    # Sidebar for settings
    st.sidebar.header("Recording Settings")
    duration = st.sidebar.slider("Recording Duration (seconds)", 1, 15, 4)

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Record Your Food")

        # Recording button
        if st.button("🎤 Start Recording", disabled=st.session_state.recording):
            if not GEMINI_API_KEY:
                st.error("Please configure Gemini API key first.")
                return

            st.session_state.recording = True

            # Record audio
            audio_data, sample_rate = record_audio(duration)

            if audio_data is not None:
                # Save audio file
                audio_file = save_audio(audio_data, sample_rate)

                if audio_file:
                    st.success("✅ Recording completed!")

                    # Play back the recording
                    st.audio(audio_file)

                    # Transcribe audio
                    with st.spinner("🔄 Transcribing audio..."):
                        try:
                            transcribed_text = transcribe_audio_modified(audio_file)

                            if transcribed_text:
                                st.success("✅ Transcription completed!")
                                st.write("**Transcribed text:**", transcribed_text)

                                # Process with Gemini
                                with st.spinner("🤖 Analyzing food items with AI..."):
                                    food_items = process_food_with_gemini(transcribed_text)

                                    if food_items:
                                        st.success("✅ Food analysis completed!")
                                        st.session_state.pending_food_items = food_items
                                    else:
                                        st.warning("No food items detected. Please try again with clearer speech.")

                                # Clean up temporary file
                                if os.path.exists(audio_file):
                                    os.remove(audio_file)
                            else:
                                st.warning("No speech detected. Please try again.")
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")

            st.session_state.recording = False

    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Entries", len(st.session_state.food_logs))
        if st.session_state.food_logs:
            total_calories = sum(item['total_calories'] for item in st.session_state.food_logs)
            st.metric("Total Calories Today", f"{total_calories:.0f}")

    # Display pending food items for confirmation
    if st.session_state.pending_food_items:
        display_food_confirmation(st.session_state.pending_food_items)

        # Clear pending items after processing
        if st.button("🗑️ Clear Pending Items"):
            st.session_state.pending_food_items = []
            st.rerun()

    # Display food logs
    st.subheader("📝 Food Log History")

    if st.session_state.food_logs:
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.food_logs)

        # Reorder columns for better display
        column_order = ['timestamp', 'food_name', 'typical_serving', 'serving_quantity',
                        'calories_per_serving', 'total_calories']
        df = df[column_order]

        st.dataframe(df, use_container_width=True)

        # Export options
        st.subheader("📤 Export Data")
        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"food_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            if st.button("🗑️ Clear All Logs"):
                st.session_state.food_logs = []
                st.success("All logs cleared!")
                st.rerun()
    else:
        st.info("No food entries yet. Start by recording your first meal!")

    # Instructions
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Configure API**: Add your Gemini API key in the sidebar or secrets
    2. **Click 'Start Recording'** to begin voice recording
    3. **Speak clearly** about what you ate (e.g., "I had a grilled chicken breast and a cup of rice")
    4. **Wait for AI analysis** - the app will identify food items and calories
    5. **Confirm items and quantities** - adjust serving sizes as needed
    6. **Add to log** - each confirmed item will be saved with calories
    7. **Export your data** as CSV when needed

    **Tips for better results:**
    - Speak clearly and mention specific foods
    - Include preparation methods (grilled, fried, etc.)
    - Be specific about food types (brown rice vs white rice)
    - Use a quiet environment for recording
    """)


def transcribe_audio_modified(filename):
    """Modified version of your transcribe_audio function"""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", compute_type="int8")

        segments, info = model.transcribe(filename)
        final_text = ""
        for segment in segments:
            final_text += segment.text

        return final_text.strip()
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None


if __name__ == "__main__":
    main()