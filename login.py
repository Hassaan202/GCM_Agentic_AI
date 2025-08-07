import streamlit as st
import pandas as pd

# Load your CSV of patient profiles
user_data = pd.read_csv("user_data/patient_profiles.csv")

# Initialize session flags
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.patient_id = None


def show_login():
    st.title("🔒 Patient Portal Login")

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            pid = st.text_input("Patient ID")
            pw = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                # Clean the data (remove whitespace)
                pid_clean = str(pid).strip()
                pw_clean = str(pw).strip()

                # Match credentials
                match = user_data[
                    (user_data["patient_id"].astype(str).str.strip() == pid_clean) &
                    (user_data["password"].astype(str).str.strip() == pw_clean)
                    ]

                if not match.empty:
                    st.session_state.logged_in = True
                    st.session_state.patient_id = pid_clean
                    st.session_state.patient_name = match.iloc[0]["name"]
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid Patient ID or Password")


# Add logout functionality for when user is logged in
def add_logout_sidebar():
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.get('patient_name', 'Patient')}")
        st.write(f"ID: {st.session_state.get('patient_id', '')}")

        if st.button("🚪 Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.patient_id = None
            st.session_state.patient_name = None
            st.rerun()


# Main logic
if not st.session_state.logged_in:
    show_login()
else:
    add_logout_sidebar()
    st.success("✅ You are logged in!")
    st.info("Please use the navigation menu to access your dashboard.")