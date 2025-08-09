import streamlit as st
from datetime import datetime
import pandas as pd
import os
import google.generativeai as genai
import json
from speech_to_text import transcribe_audio
from dotenv import load_dotenv
from pathlib import Path
from streamlit_webrtc import webrtc_streamer

load_dotenv()

# Configure Gemini API
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if 'sender_email' not in st.session_state:
    st.session_state.sender_email = ""
if "sender_app_password" not in st.session_state:
    st.session_state.sender_app_password = ""

if not st.session_state.get("logged_in", False):
    st.error("❌ Please log in to access this page")
    st.stop()

if st.session_state.google_api_key:
    genai.configure(api_key=st.session_state.google_api_key)
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
# Add session tracking for macronutrients
if 'session_carbs' not in st.session_state:
    st.session_state.session_carbs = 0.0
if 'session_fats' not in st.session_state:
    st.session_state.session_fats = 0.0
if 'session_protein' not in st.session_state:
    st.session_state.session_protein = 0.0


def record_audio():
    """Record audio for specified duration"""
    try:
        audio_data = st.audio_input("Record a voice message")
        return audio_data
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None



def process_food_with_gemini(transcribed_text):
    """Use Gemini API to extract food items with calories and macronutrients"""
    if not st.session_state.google_api_key:
        st.error("Gemini API key not configured. Please add your API key.")
        return None

    try:
        prompt = f"""
        Analyze the following text and extract food items with their typical serving sizes, calories, and macronutrients per serving.

        Text: "{transcribed_text}"

        Please respond with a JSON array where each food item is an object with these exact keys:
        - "food_name": string (name of the food item)
        - "typical_serving": string (typical serving size, e.g., "1 cup", "1 slice", "100g")
        - "calories_per_serving": number (calories in the typical serving)
        - "carbs_per_serving": number (carbohydrates in grams per serving)
        - "fats_per_serving": number (fats in grams per serving)
        - "protein_per_serving": number (protein in grams per serving)

        Example format:
        [
            {{
                "food_name": "chicken breast",
                "typical_serving": "100g",
                "calories_per_serving": 165,
                "carbs_per_serving": 0,
                "fats_per_serving": 3.6,
                "protein_per_serving": 31
            }},
            {{
                "food_name": "rice",
                "typical_serving": "1 cup cooked",
                "calories_per_serving": 205,
                "carbs_per_serving": 45,
                "fats_per_serving": 0.4,
                "protein_per_serving": 4.3
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
                st.write(f"**Carbs per serving:** {item.get('carbs_per_serving', 0)}g")
                st.write(f"**Fats per serving:** {item.get('fats_per_serving', 0)}g")
                st.write(f"**Protein per serving:** {item.get('protein_per_serving', 0)}g")

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
                total_carbs = serving_quantity * item.get('carbs_per_serving', 0)
                total_fats = serving_quantity * item.get('fats_per_serving', 0)
                total_protein = serving_quantity * item.get('protein_per_serving', 0)

                st.metric("Total Calories", f"{total_calories:.0f}")
                st.metric("Total Carbs", f"{total_carbs:.1f}g")
                st.metric("Total Fats", f"{total_fats:.1f}g")
                st.metric("Total Protein", f"{total_protein:.1f}g")

            # Add confirm button for each item
            if st.button(f"✅ Add {item['food_name']}", key=f"confirm_{i}"):
                confirmed_item = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'food_name': item['food_name'],
                    'typical_serving': item['typical_serving'],
                    'calories_per_serving': item['calories_per_serving'],
                    'carbs_per_serving': item.get('carbs_per_serving', 0),
                    'fats_per_serving': item.get('fats_per_serving', 0),
                    'protein_per_serving': item.get('protein_per_serving', 0),
                    'serving_quantity': serving_quantity,
                    'total_calories': total_calories,
                    'total_carbs': total_carbs,
                    'total_fats': total_fats,
                    'total_protein': total_protein
                }
                st.session_state.food_logs.append(confirmed_item)

                # Update session macronutrient totals
                st.session_state.session_carbs += total_carbs
                st.session_state.session_fats += total_fats
                st.session_state.session_protein += total_protein

                st.success(f"✅ Added {item['food_name']} to your food log!")

    return confirmed_items


def main():
    st.title("🍽️ Voice Food Logger with AI Analysis")
    st.markdown("Record what you eat and get automatic calorie tracking using AI!")

    # API Key configuration
    if not st.session_state.google_api_key:
        st.warning("⚠️ Gemini API key not configured. Please add your API key to use AI food analysis.")
        api_key_input = st.text_input("Enter your Gemini API Key:", type="password")
        if api_key_input:
            st.session_state.temp_api_key = api_key_input
            st.info("API key entered. Please restart the app or use secrets.toml for permanent configuration.")

    # Sidebar for settings
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.get('patient_name', 'Patient')}")
        st.write(f"ID: {st.session_state.get('patient_id', '')}")

        current_api_key = os.environ.get("GOOGLE_API_KEY", "")
        current_sender_email = st.session_state.sender_email if st.session_state.sender_email else ""
        current_sender_app_password = st.session_state.sender_app_password if st.session_state.sender_app_password else ""

        api_key = st.text_input(
            "Gemini API Key",
            value=current_api_key if current_api_key else "",
            type="password",
            help="Enter your Google Gemini API key"
        )

        # Set environment variable when key is provided
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.google_api_key = api_key
            st.success("✅ User provided API key configured")

        sender_email = st.text_input(
            "Sender Email",
            value=current_sender_email if current_sender_email else "",
            type="default",
            help="Enter your Email to use for reporting to emergency email contact"
        )

        if sender_email:
            st.session_state.sender_email = sender_email
            st.success("✅ Sender Email configured")

        sender_app_password = st.text_input(
            "Sender App Password",
            value=current_sender_app_password if current_sender_app_password else "",
            type="password",
            help="Enter your App Password from your gmail account"
        )

        if sender_app_password:
            st.session_state.sender_app_password = sender_app_password
            st.success("✅ Sender app password configured")

        if st.button("🚪 Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.patient_id = None
            st.session_state.patient_name = None
            # Clear session macronutrient tracking on logout
            st.session_state.session_carbs = 0.0
            st.session_state.session_fats = 0.0
            st.session_state.session_protein = 0.0
            st.session_state.food_logs = []
            st.session_state.google_api_key = None
            st.session_state.sender_email = None
            st.session_state.sender_app_password = None
            st.rerun()

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Record Your Food")

        st.session_state.recording = True

        AUDIO_SAVE_PATH = "audio_responses"
        os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

        # Record audio
        audio_data = record_audio()

        if audio_data is not None:
                audio_save_path = os.path.join(AUDIO_SAVE_PATH, "recorded_response.wav")
                with open(audio_save_path, "wb") as f:
                    f.write(audio_data.getbuffer())

                    # Transcribe audio
                with st.spinner("🔄 Transcribing audio..."):
                    try:
                        transcribed_text = transcribe_audio(audio_save_path)

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
                            if os.path.exists(audio_save_path):
                                os.remove(audio_save_path)
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

        # Display session macronutrient totals
        st.subheader("Session Macros")
        st.metric("Total Carbs", f"{st.session_state.session_carbs:.1f}g")
        st.metric("Total Fats", f"{st.session_state.session_fats:.1f}g")
        st.metric("Total Protein", f"{st.session_state.session_protein:.1f}g")

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
                        'calories_per_serving', 'carbs_per_serving', 'fats_per_serving', 'protein_per_serving',
                        'total_calories', 'total_carbs', 'total_fats', 'total_protein']
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
                # Reset session macronutrient totals when clearing logs
                st.session_state.session_carbs = 0.0
                st.session_state.session_fats = 0.0
                st.session_state.session_protein = 0.0
                st.success("All logs cleared!")
                st.rerun()
    else:
        st.info("No food entries yet. Start by recording your first meal!")

    # Instructions
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Configure API**: Add your Gemini API key to the secrets
    2. **Click 'Start Recording'** to begin voice recording
    3. **Speak clearly** about what you ate (e.g., "I had a grilled chicken breast and a cup of rice")
    4. **Wait for AI analysis** - the app will identify food items, calories, and macronutrients
    5. **Confirm items and quantities** - adjust serving sizes as needed
    6. **Add to log** - each confirmed item will be saved with calories and macros
    7. **Export your data** as CSV when needed

    **Tips for better results:**
    - Speak clearly and mention specific foods
    - Include preparation methods (grilled, fried, etc.)
    - Be specific about food types (brown rice vs white rice)
    - Use a quiet environment for recording
    """)


if __name__ == "__main__":
    main()