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
## Agentic Graph Structure
<img width="400" alt="agentic_graph" src="https://github.com/user-attachments/assets/a4b3ee40-9297-48cd-bba9-4acbf79b3b08" />

---
## Usage
- User ID: HUPA000XP (X = 0 to 5 inclusive)
- Password: 123

Each user has their unique profile stored in user_data/patient_profiles.csv. The associated CGM data is stored in user_data/cgm_data folder with each file identified by the user id. The CGM dataset is taken from [HUPA-UCM Diabetes Dataset](https://data.mendeley.com/datasets/3hbcscwz44/1).

Add your Gemini API key in the sidebar for testing. Sender Email and App Password (a 16 alphabet token that gives 3rd party apps access to Gmail Account) may be optionally added for emergency notifications to be sent in real-time. 

---
## Installation

```bash
# Clone the repository
git clone https://github.com/Hassaan202/GCM_Agentic_AI

# Install dependencies
pip install -r requirements.txt

#running
streamlit run main_nav.py
```

---
## Related Work
[AIDA](https://link.springer.com/article/10.1007/s44163-021-00005-1)
[D-CARE](https://www.scitepress.org/Papers/2025/132666/132666.pdf)
[Application of Chatbots to Help Patients Self-Manage Diabetes](https://pubmed.ncbi.nlm.nih.gov/39626235/)

