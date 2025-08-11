# CGM Agentic AI Application

An intelligent multi-agent system for continuous glucose monitoring (CGM) that forecasts trends, provides personalized coaching, and integrates voice-based logging — all tailored to the user’s unique profile.

---

## Features

### **1. Glucose Trend Forecasting**
- LSTM model trained on the past **6 hours** of CGM data.
- Predicts glucose levels for the next **30 minutes**.

### **2. Clinical Summarization & Management Advice**
- Generates clinical summaries from CGM data, forecasts, and user profiles.
- Offers actionable, evidence-based management suggestions for better glycemic control.

### **3. Routine Planning Agent**
- Suggests optimal daily routines (meals, sleep, exercise).
- Dynamically adapts recommendations based on glucose trends and user behavior patterns.

### **4. Emergency Reporting System**
- Detects critical CGM readings.
- Sends immediate alerts to registered emergency contacts via **SMS/Email**.

### **5. Adaptive Communication Style**
- Adjusts LLM tone and terminology based on the user’s diabetes knowledge level.
- Personalization driven by stored **user profiles**.

### **6. Voice-Based Food Logging**
- Uses **OpenAI Whisper** for speech-to-text transcription.
- Retrieves and displays macro-nutrient profiles for mentioned foods.
- Allows user confirmation of serving sizes for accurate intake logging.

### **7. Interactive Coaching Agent**
- Provides personalized answers to lifestyle, diet, glucose control, and insulin timing questions.
- Context-aware: Considers last glucose readings, CGM summaries, and recent food intake.

---

## Technology Stack

### **Backend Orchestration**
- **LangGraph**: Multi-agent workflow orchestration.

### **Large Language Models**
- **Gemini 2.5** for agentic reasoning.
- Adaptive prompt strategies based on **user profiling**.

### **Retrieval-Augmented Generation (RAG)**
- **ChromaDB** for semantic retrieval.
- **Gemini Embeddings** for medical/contextual vectorization.

### **Forecasting Engine**
- **PyTorch LSTM** model for CGM time-series forecasting.
- Hourly temporal resolution, rolling window preprocessing.

### **Voice Input System**
- **OpenAI Whisper** for speech-to-text.
- Real-time macro analysis from food database.

### **Frontend**
- **Streamlit**: Responsive web UI for visualization and interaction.
- Displays CGM trends, summaries, routine plans, and logs.

### **Emergency Contact Integration**
- Automated alerts to contacts upon abnormal CGM values.

### **User Profiling**
- Stores proficiency, preferences, and routines.
- Drives personalized recommendations and conversation tone.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Hassaan202/GCM_Agentic_AI/tree/main

# Install dependencies
pip install -r packages.txt

#running
run app.py in a python environment

