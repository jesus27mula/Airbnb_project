import streamlit as st
import pickle
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.sql_connection import fetch_top_10_services


# Exploratory Data Analysis
# =====================================

def rating_distribution(df):
    st.subheader('Ratings Distribution')

    # Filtrar dados válidos
    df_cleaned = df[df['ratings'].between(0, 5)]

    # Separar novos alojamentos (ratings = 0)
    new_properties_count = df_cleaned[df_cleaned['ratings'] == 0].shape[0]
    rated_properties = df_cleaned[df_cleaned['ratings'] > 0]

    # Definir bins personalizados
    bins = [3.5, 4.0, 4.5, 4.75, 5.0]
    labels = ['3.5-3.99', '4.0-4.49', '4.5-4.74', '4.75-5.0']

    # Categorizar ratings usando bins
    rated_properties['rating_bins'] = pd.cut(
        rated_properties['ratings'], bins=bins, labels=labels, include_lowest=True
    )

    # Combinar contagens de novos e categorizados
    count_bins = rated_properties['rating_bins'].value_counts(sort=False)
    count_bins = pd.concat(
        [pd.Series([new_properties_count], index=['0 (New)']), count_bins]
    ).reset_index()
    count_bins.columns = ['rating_range', 'count']  # Renomear colunas para compatibilidade

    # Garantir a ordem correta
    count_bins['rating_range'] = pd.Categorical(
        count_bins['rating_range'], 
        categories=['0 (New)'] + labels,  # Ordem desejada
        ordered=True
    )
    count_bins = count_bins.sort_values('rating_range')

    # Criar gráfico de barras
    rating_hist = px.bar(
        count_bins,
        x='rating_range',
        y='count',
        text='count',
        labels={'rating_range': 'Rating Ranges', 'count': 'Count'},
        title='Distribution of Ratings'
    )
    rating_hist.update_traces(
        texttemplate='%{y}', 
        textposition='outside',
        marker=dict(line=dict(width=1, color='black')),
    )
    rating_hist.update_layout(
        yaxis=dict(title='Count', type='log'),  # Escala logarítmica
        xaxis_title='Rating Ranges',
        bargap=0.2
    )

    st.plotly_chart(rating_hist, use_container_width=True, key='rating_hist')



# Función para la gráfica de dispersión de precio vs número de reseñas
def reviews_price_scatter(df):
    st.subheader('Number of Reviews vs Price per Night')
    price_hist = px.scatter(df, x='num_reviews', y='prices_per_night')
    st.plotly_chart(price_hist, use_container_width=True)

# Función para la gráfica de distribución de reseñas y valoraciones por tipo de alojamiento
def reviews_rating_distribution(df):
    st.subheader('Reviews and Ratings Distribution by Property Type')
    reviews_rating_distr = px.scatter(
        data_frame=df,
        x='ratings',
        y='num_reviews',
        log_x=True,
        color='prices_per_night',
        hover_name='property_types',
        opacity=0.7,
        size='prices_per_night',
        size_max=20,
        color_continuous_scale=px.colors.sequential.Plasma,
    )

    reviews_rating_distr.update_traces(marker=dict(line=dict(width=0.5, color="black")))

    st.plotly_chart(reviews_rating_distr, use_container_width=True)

# Price Outliers
# =====================================

# Funcion para cargar graficas con pickle
def load_and_display_pickle(file_path):
    with open(file_path, 'rb') as f:
        fig = pickle.load(f)
    st.plotly_chart(fig, use_container_width=True)


# Prices Visualizations
# =====================================



# Mapa de correlación
def correlation(df):
    st.subheader('Correlation Map')
    df_cleaned = df.drop(columns=['urls', 'timestamp', 'record_id', 'titles', 'location', 'host_name'], errors='ignore')
    df_corr = df_cleaned.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    
    st.pyplot(fig)

# Función para el gráfico de precio por tipo de propiedad
def price_property_types(df):
    st.subheader('Prices by Property Type')
    price_chart = px.box(df, x='property_types', y='prices_per_night')
    st.plotly_chart(price_chart, use_container_width=True, key='price_chart')

# Función para la gráfica de visualización entre Precio/Valoraciones por tipo de alojamiento
def price_rating_distribution(df):
    st.subheader('Price and Rating Distribution by Property Type')
    price_rating_distr_log = px.scatter(
        data_frame=df,
        x='ratings',
        y='prices_per_night',
        log_x=True,
        color='property_types',
    )
    st.plotly_chart(price_rating_distr_log, use_container_width=True)

# Función para la gráfica de precio medio según la capacidad máxima de huéspedes
def average_price_by_capacity(df):
    st.subheader('Average Price by Maximum Guest Capacity')
    avg_price = df.groupby('maximum_guests')['prices_per_night'].mean().reset_index()
    fig = px.bar(avg_price, x='maximum_guests', y='prices_per_night')
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de dispersión de precios
def price_distribution_histogram(df):
    st.subheader('Price per Night Distribution')
    fig = px.histogram(df, x='prices_per_night', nbins=80)
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    st.plotly_chart(fig, use_container_width=True)

# Feature Importance del mejor modelo
def show_feature_importance():
    with open('models/feature_importance_plotly.pkl', 'rb') as f:
        fig = pickle.load(f)
    st.plotly_chart(fig)

# Servicios
# =====================================

def top_10_services_chart():
    # Conexion con sql_connection.py
    data = fetch_top_10_services()

    df = pd.DataFrame(data, columns=["service", "count"])

    fig = px.bar(
        df,
        x="service",
        y="count",
        title="Top 10 Most Offered Services on Airbnb",
        labels={"service": "Services", "count": "Count"},
        text_auto=True,
        color="service"
    )
    st.plotly_chart(fig, use_container_width=True)   














