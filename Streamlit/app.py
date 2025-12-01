import streamlit as st
import pandas as pd
from pages_section import landing_page, data_analysis, machine_learning, database, about


# Cargar el CSV en session_state
if 'df' not in st.session_state:
    file_path = 'data/df_final_cleaned.csv'# Dataframe procesado para visualizaciones
    df = pd.read_csv(file_path)
    # Almacenar el DataFrame limpio en session_state
    st.session_state.df = df

if 'df_processed' not in st.session_state:
    df_ml = pd.read_csv('data/df_processed_to_ML.csv')  # DataFrame procesado para modelos
    st.session_state.df_processed = df_ml

if 'df_sentiment' not in st.session_state:
    df_sentiment = pd.read_csv('data/df_rec_st.csv')  # DataFrame para análise de sentimientos
    st.session_state.df_sentiment = df_sentiment

# Acceder al DataFrame directamente desde session_state
df = st.session_state.df
df_processed = st.session_state.df_processed
df_sentiment = st.session_state.df_sentiment

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