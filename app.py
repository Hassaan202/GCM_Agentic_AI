import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import glob
import os
from langchain_core.messages import ToolMessage, HumanMessage
from sklearn.preprocessing import MinMaxScaler
from graph import graph
from langgraph.types import Command, interrupt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
if 'current_patient_id' not in st.session_state:
    st.session_state.current_patient_id = None

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
low_range = st.sidebar.number_input("Low Glucose Range (mg/dL)", value=70, min_value=50, max_value=100)
high_range = st.sidebar.number_input("High Glucose Range (mg/dL)", value=180, min_value=150, max_value=250)

# File upload section
st.header("📁 Data Upload")
upload_option = st.radio("Choose data source:", ["Upload CSV File", "Load from Directory"])

cgm_df = None

if upload_option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            # Try semicolon delimiter first (as in main.py)
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

            # Handle patient ID column - check for both 'patient_id' and 'id'
            patient_id_status = 1  # 1 = present, -1 = missing
            if 'patient_id' in df.columns:
                df['patient_id'] = df['patient_id']
            elif 'id' in df.columns:
                df['patient_id'] = df['id']
            else:
                # Continue with default patient_id if other required columns are present
                patient_id_status = -1
                df['patient_id'] = 'default_patient'
                st.warning("⚠️ Warning: No 'patient_id' or 'id' column found. Using default patient ID.")

            # Handle glucose column - check for both 'glucose' and 'gl'
            if 'glucose' in df.columns:
                df['glucose'] = df['glucose']
            elif 'gl' in df.columns:
                df['glucose'] = df['gl']
            else:
                st.error("❌ Error: CSV must contain either a 'glucose' or 'gl' column")
                st.stop()

            cgm_df = df
            # Store patient_id_status for later use
            st.session_state.patient_id_status = patient_id_status
            st.success("✅ Data loaded successfully!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

elif upload_option == "Load from Directory":
    data_path = st.text_input("Enter path to preprocessed data directory:",
                              value="HUPA_UCM_Dataset/Preprocessed")

    if st.button("Load Directory Data") and data_path:
        try:
            csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))

            if not csv_files:
                st.error("No CSV files found in the specified directory")
            else:
                all_data = []
                progress_bar = st.progress(0)

                for i, file in enumerate(csv_files):
                    patient_id = os.path.basename(file).replace(".csv", "")

                    try:
                        # Read with semicolon delimiter (as in main.py)
                        df = pd.read_csv(file, delimiter=';', parse_dates=['time'])

                        # Validate required columns
                        if 'time' not in df.columns:
                            st.error(f"❌ Error in {file}: Missing 'time' column")
                            continue

                        # Handle patient ID column
                        if 'patient_id' not in df.columns and 'id' not in df.columns:
                            # Use filename as patient_id if neither column exists
                            df['patient_id'] = patient_id
                        elif 'id' in df.columns and 'patient_id' not in df.columns:
                            df['patient_id'] = df['id']

                        # Handle glucose column
                        if 'glucose' not in df.columns and 'gl' not in df.columns:
                            st.error(f"❌ Error in {file}: Missing 'glucose' or 'gl' column")
                            continue
                        elif 'gl' in df.columns and 'glucose' not in df.columns:
                            df['glucose'] = df['gl']

                        all_data.append(df)

                    except Exception as e:
                        st.error(f"❌ Error reading {file}: {e}")
                        continue

                    progress_bar.progress((i + 1) / len(csv_files))

                # Combine all into one DataFrame
                if all_data:
                    cgm_df = pd.concat(all_data, ignore_index=True)
                    # For directory loading, patient_id is always available (from filename)
                    st.session_state.patient_id_status = 1
                    st.success(f"✅ Loaded {len(all_data)} patient files successfully!")
                else:
                    st.error("❌ No valid patient files could be loaded")

        except Exception as e:
            st.error(f"Error loading directory: {e}")

# Main analysis section
if cgm_df is not None:
    st.header("👥 Patient Analysis")

    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", cgm_df['patient_id'].nunique())
    with col2:
        st.metric("Total Records", len(cgm_df))
    with col3:
        st.metric("Date Range", f"{cgm_df['time'].min().date()} to {cgm_df['time'].max().date()}")

    # Patient selection
    patient_ids = cgm_df['patient_id'].unique()
    selected_patient = st.selectbox("Select a Patient ID", patient_ids)

    if selected_patient:
        # Filter patient data
        patient_data = cgm_df[cgm_df['patient_id'] == selected_patient].sort_values('time')

        # Reset analysis state if patient changed
        if st.session_state.current_patient_id != selected_patient:
            st.session_state.analysis_state = 'idle'
            st.session_state.analysis_result = None
            st.session_state.interrupt_question = None
            st.session_state.graph_config = None
            st.session_state.current_patient_id = selected_patient

        st.session_state.current_patient_data = patient_data

        # Display patient summary
        st.subheader(f"📊 Patient Summary: {selected_patient}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(patient_data))
        with col2:
            if 'glucose' in patient_data.columns:
                st.metric("Avg Glucose", f"{patient_data['glucose'].mean():.1f} mg/dL")
        with col3:
            st.metric("Date Range", f"{patient_data['time'].nunique()} days")
        with col4:
            if 'glucose' in patient_data.columns:
                last_glucose = patient_data['glucose'].iloc[-1]
                st.metric("Last Reading", f"{last_glucose:.1f} mg/dL")

        # Data visualization
        fig = go.Figure()
        if 'glucose' in patient_data.columns:
            st.subheader("📈 Glucose Trends")

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
                title="Glucose Levels Over Time",
                xaxis_title="Time",
                yaxis_title="Glucose (mg/dL)",
                height=400
            )

        if 'glucose' in patient_data.columns:
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

            st.subheader("⏱️ Time in Range Stats")

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

                        # Run the graph analysis
                        result = graph.invoke({
                            "patient_id": selected_patient,
                            "input_tensor": input_tensor.tolist(),
                            "raw_patient_data": patient_data.to_dict(orient="records"),
                            "low_range": low_range,
                            "high_range": high_range,
                            "messages": [],
                            "rag_complete": False,
                        }, config)

                        # Handle interrupts (user input required)
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
                st.subheader("📈 Glucose Forecast")

                last_time = patient_data['time'].iloc[-1]
                future_time = last_time + pd.Timedelta(minutes=5)

                # Create new figure for forecast
                forecast_fig = go.Figure()

                # Add historical data
                forecast_fig.add_trace(go.Scatter(
                    x=patient_data['time'],
                    y=patient_data['glucose'],
                    mode='lines+markers',
                    name='Glucose Level',
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
                    title="Glucose Levels with Forecast",
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

            # Reset button to run again
            if st.button("🔄 Run New Analysis"):
                st.session_state.analysis_state = 'idle'
                st.session_state.analysis_result = None
                st.session_state.interrupt_question = None
                st.session_state.graph_config = None
                st.rerun()

else:
    st.info("👆 Please upload data or specify a directory path to begin analysis")

# Footer
st.markdown("---")
st.markdown("*Advanced Glucose Monitoring System powered by AI and Graph-based Analysis*")