import streamlit as st
import pandas as pd
from utils import init_session_state
from pages_section import landing_page, data_analysis, machine_learning, database, about

# Inicializar session_state con utils
init_session_state()

# Acceder a los DataFrames
df = st.session_state.df
df_processed = st.session_state.df_processed
df_sentiment = st.session_state.df_sentiment



# Cargar el CSV en session_state
from utils import load_csv, load_image, load_analysis_image, get_data_path, get_images_path

page = st.radio(
    "**🏠 Dear User, choose what you want to discover! 🏠**",
    ('Home', 'Data Analysis', 'Machine Learning', 'Database', 'About'),
    horizontal=True
)

if page == 'Home':
    landing_page.show()
elif page == 'Data Analysis':
    data_analysis.show(df)
elif page == 'Machine Learning':
    machine_learning.show()
elif page == 'Database':
    database.show()    
elif page == 'About':
    about.show()