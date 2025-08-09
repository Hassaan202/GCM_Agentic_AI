import streamlit as st
import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from graph import graph
from langgraph.types import Command
import plotly.express as px
import plotly.graph_objects as go


if not st.session_state.get("logged_in", False):
    st.error("❌ Please log in to access this page")
    st.stop()

if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if 'sender_email' not in st.session_state:
    st.session_state.sender_email = ""
if "sender_app_password" not in st.session_state:
    st.session_state.sender_app_password = ""


# ------------------------
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------

# Page configuration
st.set_page_config(
    page_title="🩺 Advanced Glucose Level Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Advanced Glucose Level Predictor")
st.markdown("*Using AI-powered graph-based analysis for comprehensive glucose monitoring*")

# Initialize session state for analysis management
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = 'idle'  # 'idle', 'running', 'waiting_input', 'complete'
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'interrupt_question' not in st.session_state:
    st.session_state.interrupt_question = None
if 'graph_config' not in st.session_state:
    st.session_state.graph_config = None
if 'current_patient_data' not in st.session_state:
    st.session_state.current_patient_data = None

# Add food log session state tracking
if 'session_carbs' not in st.session_state:
    st.session_state.session_carbs = 0.0
if 'session_fats' not in st.session_state:
    st.session_state.session_fats = 0.0
if 'session_protein' not in st.session_state:
    st.session_state.session_protein = 0.0
if 'food_logs' not in st.session_state:
    st.session_state.food_logs = []

# Sidebar for configuration
with (((st.sidebar))):
    st.write(f"👤 Welcome, {st.session_state.get('patient_name', 'Patient')}")
    st.write(f"ID: {st.session_state.get('patient_id', '')}")

    st.header("⚙️ Configuration")
    low_range = st.number_input("Low Glucose Range (mg/dL)", value=70, min_value=50, max_value=100)
    high_range = st.number_input("High Glucose Range (mg/dL)", value=180, min_value=150, max_value=250)

    # Add this after the existing sidebar metrics
    st.header("🍽️ Today's Nutrition")
    st.metric("Carbs", f"{st.session_state.session_carbs:.1f}g")
    st.metric("Protein", f"{st.session_state.session_protein:.1f}g")
    st.metric("Fats", f"{st.session_state.session_fats:.1f}g")
    if st.session_state.food_logs:
        total_calories = sum(item['total_calories'] for item in st.session_state.food_logs)
        st.metric("Total Calories", f"{total_calories:.0f}")

    st.header("🔑 API Configuration")

    # Check if API key is already set
    current_api_key = os.environ.get("GOOGLE_API_KEY", "")
    current_sender_email = st.session_state.sender_email if "sender_email" in st.session_state else ""
    current_sender_app_password = st.session_state.sender_app_password if "sender_app_password" in st.session_state else ""

    # API key input
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
        st.session_state.session_carbs = 0.0
        st.session_state.session_fats = 0.0
        st.session_state.session_protein = 0.0
        st.session_state.food_logs = []
        st.session_state.google_api_key = None
        st.session_state.sender_email = None
        st.session_state.sender_app_password = None
        st.rerun()

# Get current user's patient ID
current_patient_id = st.session_state.get('patient_id', '')

def check_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ Please configure your Gemini API key in the sidebar")
        st.stop()

df = ""

def get_patient_data(patient_id, dataframe):
    record = dataframe[dataframe["patient_id"] == patient_id]
    if record.empty:
        return None
    return record.iloc[0]  # return as a Series (single row)

# Data loading section
st.header("📁 Your Data")

cgm_df = None
data_found = False

# Try to load user's specific data file
if current_patient_id:
    data_path = f"user_data/cgm_data/{current_patient_id}.csv"

    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, delimiter=';')

            # Convert time column
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            else:
                st.error("❌ Error: Your data file is missing the 'time' column")
                st.stop()

            # Set patient_id to current user
            # df['patient_id'] = current_patient_id

            # Handle glucose column - check for both 'glucose' and 'gl'
            if 'glucose' in df.columns:
                df['glucose'] = df['glucose']
            elif 'gl' in df.columns:
                df['glucose'] = df['gl']
            else:
                st.error("❌ Error: Your data file must contain either a 'glucose' or 'gl' column")
                st.stop()

            cgm_df = df
            data_found = True
            st.success(f"✅ Your glucose data loaded successfully! ({len(df)} records)")

            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(cgm_df))
            with col2:
                st.metric("Date Range", f"{cgm_df['time'].min().date()} to {cgm_df['time'].max().date()}")
            with col3:
                st.metric("Days of Data", cgm_df['time'].dt.date.nunique())

        except Exception as e:
            st.error(f"❌ Error loading your data file: {e}")
    else:
        st.warning(f"⚠️ No data file found for your patient ID: {current_patient_id}")

# File upload as fallback option
if not data_found:
    st.info("💡 You can upload your glucose data file below:")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            # Try semicolon delimiter first
            df = pd.read_csv(uploaded_file, delimiter=';')
            if len(df.columns) == 1:
                # If semicolon didn't work, try comma
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

            # Convert time column
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            else:
                st.error("❌ Error: CSV must contain a 'time' column")
                st.stop()

            # Set patient_id to current user
            df['patient_id'] = current_patient_id

            # Handle glucose column - check for both 'glucose' and 'gl'
            if 'glucose' in df.columns:
                df['glucose'] = df['glucose']
            elif 'gl' in df.columns:
                df['glucose'] = df['gl']
            else:
                st.error("❌ Error: CSV must contain either a 'glucose' or 'gl' column")
                st.stop()

            cgm_df = df
            data_found = True
            st.success("✅ Data uploaded successfully!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Main analysis section - only show if data is available
if cgm_df is not None and data_found:
    # Use the patient data directly (no patient selection needed)
    patient_data = cgm_df.sort_values('time')
    st.session_state.current_patient_data = patient_data

    # Display patient summary
    st.subheader(f"📊 Your Glucose Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(patient_data))
    with col2:
        if 'glucose' in patient_data.columns:
            st.metric("Avg Glucose", f"{patient_data['glucose'].mean():.1f} mg/dL")
    with col3:
        st.metric("Date Range", f"{patient_data['time'].dt.date.nunique()} days")
    with col4:
        if 'glucose' in patient_data.columns:
            last_glucose = patient_data['glucose'].iloc[-1]
            st.metric("Last Reading", f"{last_glucose:.1f} mg/dL")

    # Data visualization
    if 'glucose' in patient_data.columns:
        st.subheader("📈 Your Glucose Trends")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=patient_data['time'],
            y=patient_data['glucose'],
            mode='lines+markers',
            name='Glucose Level',
            line=dict(color='blue', width=2)
        ))

        # Add reference lines
        fig.add_hline(y=low_range, line_dash="dash", line_color="red",
                      annotation_text=f"Low ({low_range})")
        fig.add_hline(y=high_range, line_dash="dash", line_color="red",
                      annotation_text=f"High ({high_range})")

        fig.update_layout(
            title="Your Glucose Levels Over Time",
            xaxis_title="Time",
            yaxis_title="Glucose (mg/dL)",
            height=400
        )

        # Calculate stats
        total = len(patient_data)
        in_range = ((patient_data['glucose'] >= low_range) &
                    (patient_data['glucose'] <= high_range)).sum()
        below_range = (patient_data['glucose'] < low_range).sum()
        above_range = (patient_data['glucose'] > high_range).sum()

        tir = (in_range / total) * 100
        below_pct = (below_range / total) * 100
        above_pct = (above_range / total) * 100

        # Display chart
        st.plotly_chart(fig, use_container_width=True, key='glucose_chart')

        st.subheader("⏱️ Your Time in Range Stats")

        # Layout: Pie chart (3/4) + Stats (1/4)
        left, right = st.columns([3, 1])

        # --- Left: Pie Chart ---
        pie_data = {
            'Category': ['In Range', 'Below Range', 'Above Range'],
            'Value': [in_range, below_range, above_range]
        }
        pie_fig = px.pie(pie_data, names='Category', values='Value',
                         color='Category',
                         color_discrete_map={
                             'In Range': 'green',
                             'Below Range': 'blue',
                             'Above Range': 'red'
                         },
                         hole=0.4)
        pie_fig.update_traces(textinfo='percent+label')
        left.plotly_chart(pie_fig, use_container_width=True, key='pie_chart')

        # --- Right: Numeric Stats ---
        right.metric("In Range", f"{tir:.1f}%")
        right.metric("Below Range", f"{below_pct:.1f}%")
        right.metric("Above Range", f"{above_pct:.1f}%")

    # Prediction section
    st.subheader("🔮 AI-Powered Prediction & Analysis")

    # Analysis state management
    if st.session_state.analysis_state == 'idle':
        if st.button("🚀 Run Advanced Analysis", type="primary"):
            # check_api_key()
            st.session_state.analysis_state = 'running'
            st.rerun()

    elif st.session_state.analysis_state == 'running':
        try:
            with st.spinner("Running AI analysis..."):
                # Prepare features
                features = ['glucose', 'calories', 'heart_rate', 'steps',
                            'basal_rate', 'bolus_volume_delivered', 'carb_input']

                # Check which features are available
                available_features = [f for f in features if f in patient_data.columns]

                if not available_features:
                    st.error("No required features found in the data")
                    st.session_state.analysis_state = 'idle'
                else:
                    # Extract relevant features
                    data = patient_data[available_features].ffill().bfill()

                    # Fit scaler on full history for normalization
                    scaler = MinMaxScaler()
                    data_scaled = scaler.fit_transform(data)

                    # Get last 72 time steps (6 hours) for prediction
                    window_size = min(72, len(data_scaled))
                    window = data_scaled[-window_size:]
                    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

                    # Graph configuration
                    config = {"configurable": {"thread_id": "1"}}
                    st.session_state.graph_config = config

                    # patient data loaded
                    patient_bio_df = pd.read_csv("user_data/patient_profiles.csv")
                    patient_bio_data = get_patient_data(current_patient_id, patient_bio_df) # the user data row

                    graph_params = {
                        "patient_id": current_patient_id,
                        "input_tensor": input_tensor.tolist(),
                        "raw_patient_data": patient_data.to_dict(orient="records"),
                        "low_range": low_range,
                        "high_range": high_range,
                        "messages": [],
                        "rag_complete": False,
                        "age": patient_bio_data["age"].item(),
                        "gender": patient_bio_data["gender"],
                        "diabetes_proficiency": patient_bio_data["Diabetes Proficiency"],
                        "emergency_contact_number": patient_bio_data["emergency_contact_number"],
                        "emergency_email": patient_bio_data["emergency_email"],
                        "name": patient_bio_data["name"],
                        "id": patient_bio_data["patient_id"],
                        "carbs_grams": st.session_state.session_carbs,
                        "protein_grams": st.session_state.session_protein,
                        "fat_grams": st.session_state.session_fats,
                        "food_logs": st.session_state.food_logs
                    }

                    if current_sender_email and current_sender_app_password:
                        graph_params.update({
                            "sender_email": current_sender_email,
                            "sender_account_app_password": current_sender_app_password
                        })
                    else:
                        graph_params.update({
                            "sender_email": None,
                            "sender_account_app_password": None
                        })

                    # Run the graph analysis
                    result = graph.invoke(graph_params, config)

                    # Handle user input
                    interrupts = result.get("__interrupt__", [])
                    if interrupts:
                        interrupt_value = interrupts[0].value
                        question = interrupt_value.get("question")

                        st.session_state.interrupt_question = question
                        st.session_state.analysis_result = result
                        st.session_state.analysis_state = 'waiting_input'
                        st.rerun()
                    else:
                        # No interrupts, analysis complete
                        st.session_state.analysis_result = result
                        st.session_state.analysis_state = 'complete'
                        st.rerun()

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)
            st.session_state.analysis_state = 'idle'

    elif st.session_state.analysis_state == 'waiting_input':
        st.warning("⚠️ Additional Information Required")
        user_input = st.text_input(st.session_state.interrupt_question, key="user_interrupt_input")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Submit Response", type="primary"):
                if user_input:
                    try:
                        with st.spinner("Processing your response..."):
                            result = graph.invoke(Command(resume=user_input), config=st.session_state.graph_config)
                            st.session_state.analysis_result = result
                            st.session_state.analysis_state = 'complete'
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to process response: {e}")
                        st.exception(e)
                else:
                    st.error("Please provide a response before submitting")

        with col2:
            if st.button("Cancel Analysis"):
                st.session_state.analysis_state = 'idle'
                st.session_state.analysis_result = None
                st.session_state.interrupt_question = None
                st.session_state.graph_config = None
                st.rerun()

    elif st.session_state.analysis_state == 'complete':
        # Display results
        result = st.session_state.analysis_result
        st.success("✅ Analysis Complete!")

        # Main results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔮 Prediction Results")
            if "predicted_glucose" in result:
                st.metric("Predicted Glucose", f"{result['predicted_glucose']:.1f} mg/dL")
            if "glucose_level" in result:
                level = result["glucose_level"]
                color = "green" if level == "Normal" else "orange" if level == "Warning" else "red"
                st.markdown(f"**Glucose Level:** :{color}[{level}]")
            if "trend_note" in result:
                st.info(f"📈 **Trend:** {result['trend_note']}")

        with col2:
            st.subheader("🚨 Emergency Status")
            if result.get("emergency", False):
                st.error("🚨 EMERGENCY DETECTED")
            else:
                st.success("✅ No Emergency Detected")

        # Forecasting graph section
        if 'glucose' in patient_data.columns and "predicted_glucose" in result:
            st.subheader("📈 Your Glucose Forecast")

            last_time = patient_data['time'].iloc[-1]
            future_time = last_time + pd.Timedelta(minutes=5)

            # Create new figure for forecast
            forecast_fig = go.Figure()

            # Add historical data
            forecast_fig.add_trace(go.Scatter(
                x=patient_data['time'],
                y=patient_data['glucose'],
                mode='lines+markers',
                name='Your Glucose Level',
                line=dict(color='blue', width=2)
            ))

            # Line connecting last point to predicted
            forecast_fig.add_trace(go.Scatter(
                x=[last_time, future_time],
                y=[patient_data['glucose'].iloc[-1], result['predicted_glucose']],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='orange', dash='dot', width=2),
                marker=dict(color='orange', size=10)
            ))

            # Reference lines
            forecast_fig.add_hline(y=low_range, line_dash="dash", line_color="red",
                                   annotation_text=f"Low ({low_range})")
            forecast_fig.add_hline(y=high_range, line_dash="dash", line_color="red",
                                   annotation_text=f"High ({high_range})")

            forecast_fig.update_layout(
                title="Your Glucose Levels with Forecast",
                xaxis_title="Time",
                yaxis_title="Glucose (mg/dL)",
                height=400
            )

            # Focus on last few hours
            start_zoom = last_time - pd.Timedelta(hours=3)
            end_zoom = future_time + pd.Timedelta(minutes=15)
            forecast_fig.update_xaxes(range=[start_zoom, end_zoom])

            st.plotly_chart(forecast_fig, use_container_width=True, key="forecast_chart")

        # Advice section
        if "advice" in result:
            st.subheader("🧠 AI Recommendations")
            st.markdown(result["advice"])

        # Routine Planning section
        if "routine_plan" in result:
            st.subheader("📅 Personalized Routine Plan")
            st.markdown(result["routine_plan"])

        # Reset button to run again
        if st.button("🔄 Run New Analysis"):
            st.session_state.analysis_state = 'idle'
            st.session_state.analysis_result = None
            st.session_state.interrupt_question = None
            st.session_state.graph_config = None
            st.rerun()

else:
    if current_patient_id:
        st.info(
            f"👆 Please upload your glucose data file or ensure your data file exists at: `user_data/cgm_data/{current_patient_id}.csv`")
    else:
        st.error("❌ No patient ID found. Please log in again.")

# Footer
st.markdown("---")
st.markdown("*Your Personal Advanced Glucose Monitoring System powered by AI*")