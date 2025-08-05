import streamlit as st
import pandas as pd
import os
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini AI: {e}")
    st.stop()

# Check authentication
if not st.session_state.get("logged_in", False):
    st.error("❌ Please log in to access the chatbot")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 Glucose Monitoring Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Goal-Oriented Coaching Agent")
st.markdown("*AI-powered support for your diabetes management journey*")

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Initialize system context flag
if "context_initialized" not in st.session_state:
    st.session_state.context_initialized = False

def get_patient_context():
    """Generate enhanced system prompt based on user's session data"""

    patient_id = st.session_state.get('patient_id', 'Unknown')
    patient_name = st.session_state.get('patient_name', 'Patient')

    # Get nutrition data
    session_carbs = st.session_state.get('session_carbs', 0.0)
    session_protein = st.session_state.get('session_protein', 0.0)
    session_fats = st.session_state.get('session_fats', 0.0)
    food_logs = st.session_state.get('food_logs', [])

    # Calculate total calories if food logs exist
    total_calories = sum(item.get('total_calories', 0) for item in food_logs) if food_logs else 0

    # Get current patient data if available
    current_patient_data = st.session_state.get('current_patient_data', None)
    glucose_summary = ""

    if current_patient_data is not None and not current_patient_data.empty:
        try:
            latest_glucose = current_patient_data['glucose'].iloc[
                -1] if 'glucose' in current_patient_data.columns else None
            avg_glucose = current_patient_data['glucose'].mean() if 'glucose' in current_patient_data.columns else None
            total_records = len(current_patient_data)
            date_range = current_patient_data[
                'time'].dt.date.nunique() if 'time' in current_patient_data.columns else None

            glucose_summary = f"""
            Recent Glucose Data:
            - Latest Reading: {latest_glucose:.1f} mg/dL
            - Average Glucose: {avg_glucose:.1f} mg/dL
            - Total Records: {total_records}
            - Days of Data: {date_range}
            """
        except Exception:
            glucose_summary = "Glucose data available but unable to process summary."

    # Get analysis results if available
    analysis_result = st.session_state.get('analysis_result', None)
    analysis_summary = ""

    if analysis_result:
        predicted_glucose = analysis_result.get('predicted_glucose', 'N/A')
        glucose_level = analysis_result.get('glucose_level', 'N/A')
        emergency_status = "Emergency Detected" if analysis_result.get('emergency', False) else "No Emergency"

        analysis_summary = f"""
        Latest AI Analysis:
        - Predicted Glucose: {predicted_glucose} mg/dL
        - Glucose Level Status: {glucose_level}
        - Emergency Status: {emergency_status}
        """

    # Try to load patient bio data
    bio_data_summary = ""
    try:
        if os.path.exists("user_data/patient_profiles.csv"):
            patient_bio_df = pd.read_csv("user_data/patient_profiles.csv")
            patient_bio = patient_bio_df[patient_bio_df["patient_id"] == patient_id]
            if not patient_bio.empty:
                bio_data = patient_bio.iloc[0]
                bio_data_summary = f"""
                Patient Profile:
                - Age: {bio_data.get('age', 'N/A')}
                - Gender: {bio_data.get('gender', 'N/A')}
                - Diabetes Proficiency: {bio_data.get('Diabetes Proficiency', 'N/A')}
                """
    except Exception:
        bio_data_summary = "Patient profile data unavailable."

    # Food logs summary
    food_summary = ""
    if food_logs:
        recent_foods = [item.get('food_name', 'Unknown') for item in food_logs[-3:]]  # Last 3 items
        food_summary = f"""
        Recent Food Intake:
        - Foods: {', '.join(recent_foods)}
        """

    system_prompt = f"""
    You are a specialized AI assistant for diabetes and glucose monitoring. You are helping {patient_name} (Patient ID: {patient_id}).

    PATIENT CONTEXT:
    {bio_data_summary}

    {glucose_summary}

    {analysis_summary}

    TODAY'S NUTRITION TRACKING:
    - Carbohydrates: {session_carbs:.1f}g
    - Protein: {session_protein:.1f}g
    - Fats: {session_fats:.1f}g
    - Total Calories: {total_calories:.0f}

    {food_summary}

    INSTRUCTIONS:
    1. Provide personalized advice based on the patient's current glucose levels, nutrition intake, and profile
    2. Focus on practical, actionable guidance for diabetes management
    3. Consider the patient's diabetes proficiency level when explaining concepts
    4. Be supportive and encouraging while being medically accurate
    5. If glucose levels indicate concern, provide appropriate guidance but remind them to consult healthcare providers for emergencies
    6. Help interpret their glucose trends and patterns
    7. Suggest lifestyle modifications, meal planning, or timing recommendations when appropriate
    8. Answer questions about diabetes management, glucose monitoring, nutrition, and related topics

    IMPORTANT LIMITATIONS:
    - You are not a replacement for medical professional advice
    - For emergencies or severe glucose episodes, always recommend immediate medical attention
    - Do not provide specific medication dosing advice
    - Encourage regular consultation with healthcare providers

    Maintain a friendly, supportive, and professional tone throughout the conversation. Keep responses slightly brief.
    """

    return system_prompt


def get_gemini_response(user_input, context):
    """Get response from Gemini AI with context"""
    try:
        # Combine context with user input
        full_prompt = f"{context}\n\nUser Question: {user_input}"

        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}"


# Sidebar with patient info and quick actions
with st.sidebar:
    st.write(f"👤 **{st.session_state.get('patient_name', 'Patient')}**")
    st.write(f"ID: {st.session_state.get('patient_id', '')}")

    st.header("📊 Quick Stats")

    # Nutrition summary
    st.subheader("🍽️ Today's Nutrition")
    st.metric("Carbs", f"{st.session_state.get('session_carbs', 0):.1f}g")
    st.metric("Protein", f"{st.session_state.get('session_protein', 0):.1f}g")
    st.metric("Fats", f"{st.session_state.get('session_fats', 0):.1f}g")

    if st.session_state.get('food_logs', []):
        total_calories = sum(item.get('total_calories', 0) for item in st.session_state.food_logs)
        st.metric("Total Calories", f"{total_calories:.0f}")

    # Latest glucose info if available
    current_data = st.session_state.get('current_patient_data', None)
    if current_data is not None and not current_data.empty and 'glucose' in current_data.columns:
        st.subheader("🩸 Latest Glucose")
        latest_glucose = current_data['glucose'].iloc[-1]
        st.metric("Current Level", f"{latest_glucose:.1f} mg/dL")

    # Analysis status
    if st.session_state.get('analysis_result'):
        st.subheader("🔮 AI Analysis")
        result = st.session_state.analysis_result
        if 'predicted_glucose' in result:
            st.metric("Predicted", f"{result['predicted_glucose']:.1f} mg/dL")
        if 'glucose_level' in result:
            level = result['glucose_level']
            color = "🟢" if level == "Normal" else "🟡" if level == "Warning" else "🔴"
            st.write(f"Status: {color} {level}")

    st.markdown("---")

    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    if st.button("📈 View Main Dashboard"):
        st.switch_page("app.py")

    if st.button("🍽️ Food Logger"):
        st.switch_page("pages/food_log.py")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.patient_id = None
        st.session_state.patient_name = None
        st.session_state.session_carbs = 0.0
        st.session_state.session_fats = 0.0
        st.session_state.session_protein = 0.0
        st.session_state.food_logs = []
        st.session_state.chat_messages = []
        st.switch_page("login.py")

# Main chat interface
st.header("💬 Chat with Your Assistant")

# Initialize context message
if not st.session_state.context_initialized:
    context = get_patient_context()
    welcome_message = "Hello! I'm your personal glucose monitoring assistant. I have access to your current glucose data, nutrition tracking, and recent analysis results. How can I help you manage your diabetes today?"

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": welcome_message,
        "timestamp": datetime.now()
    })
    st.session_state.context_initialized = True

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show timestamp for messages
        if "timestamp" in message:
            st.caption(f"🕒 {message['timestamp'].strftime('%H:%M:%S')}")

# Chat input
if prompt := st.chat_input("Ask me anything about your glucose management..."):
    # Add user message to chat history
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now()
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = get_patient_context()
            response = get_gemini_response(prompt, context)
            st.markdown(response)
            st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")

    # Add assistant response to chat history
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now()
    })

# Chat management
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.context_initialized = False
        st.rerun()

with col2:
    if st.button("🔄 Refresh Context"):
        st.session_state.context_initialized = False
        st.rerun()

with col3:
    # Export chat option
    if st.session_state.chat_messages and st.button("💾 Export Chat"):
        chat_export = []
        for msg in st.session_state.chat_messages:
            chat_export.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", datetime.now()).isoformat()
            })

        chat_df = pd.DataFrame(chat_export)
        csv = chat_df.to_csv(index=False)

        st.download_button(
            label="📋 Download Chat History",
            data=csv,
            file_name=f"glucose_chat_{st.session_state.get('patient_id', 'user')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Usage tips
with st.expander("💡 How to use this chatbot"):
    st.markdown("""
    **Your AI assistant can help with:**

    🩸 **Glucose Management**
    - Interpret your current glucose readings
    - Explain glucose trends and patterns
    - Provide guidance on managing highs and lows

    🍽️ **Nutrition Guidance**
    - Analyze your daily nutrition intake
    - Suggest meal timing and composition
    - Help with carb counting

    📊 **Data Analysis**
    - Explain your AI analysis results
    - Help understand predictions
    - Discuss time-in-range statistics

    🏥 **General Diabetes Care**
    - Answer questions about diabetes management
    - Provide lifestyle recommendations
    - Explain medical terms and concepts

    **Example questions:**
    - "What does my latest glucose reading mean?"
    - "Should I be concerned about my carb intake today?"
    - "How can I improve my time in range?"
    - "What should I do if my glucose is trending high?"

    **Remember:** This assistant provides educational support but is not a replacement for professional medical advice.
    """)

# Footer
st.markdown("---")
st.markdown(
    "*🤖 AI-Powered Diabetes Management Support | Always consult your healthcare provider for medical decisions*")