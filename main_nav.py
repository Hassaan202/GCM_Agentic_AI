import streamlit as st

analysis_page = st.Page("app.py", title="Analysis")
food_log_page = st.Page("food_logging.py", title="Food Logging")

pg = st.navigation([analysis_page, food_log_page])
st.set_page_config(page_title="Home")
pg.run()