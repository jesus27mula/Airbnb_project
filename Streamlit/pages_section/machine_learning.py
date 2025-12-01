import streamlit as st
import joblib
import numpy as np
import pandas as pd
from modules.visualizations import show_feature_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Funciones auxiliares para cargar los diferentes modelos y escaladores
@st.cache_resource
def load_scalers():
    x_scaler = joblib.load('models/x_scaler.pkl')
    y_scaler = joblib.load('models/y_scaler.pkl')
    return x_scaler, y_scaler

@st.cache_resource
def load_models():
    return {
        "lightgbm": joblib.load("models/best_lightgbm_model.pkl"),
        "neural_network": joblib.load("models/simple_nn_model.pkl"),
        "recommendation": joblib.load("models/NearestNeighbors.pkl"),
    }

def show_price_prediction(df_processed):
    st.header("💸 Price Prediction Model")
    st.write("Using this LightGBM model, we will predict the price of a property based on its features.")

    # Cargar los modelos y el KNN imputer
    models = load_models()
    
    model = models["lightgbm"]["model"]
    
    # Cargar el KNN Imputer
    knn_imputer = joblib.load("models/knn_imputer.pkl")

    # Entradas del usuario
    st.subheader("Please enter the following details about the property:")

    max_guests = st.sidebar.selectbox("Maximum Guests", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
    bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    beds = st.sidebar.selectbox("Number of Beds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5], index=1)
    cleaning_fee = st.sidebar.slider("Cleaning Fee (€)", min_value=0, value=50)

    # Crear un DataFrame con las entradas del usuario
    user_input = {
        "maximum_guests": max_guests,
        "dormitorios": bedrooms,
        "camas": beds,
        "baños": bathrooms,
        "cleaning_fee": cleaning_fee
    }

    input_df = pd.DataFrame(user_input, index=[0])

    # Rellenar las columnas restantes con el KNN Imputer (sin incluir 'prices_per_night')
    columns_to_impute = [col for col in df_processed.columns if col not in user_input and col != "prices_per_night"]

    for col in columns_to_impute:
        input_df[col] = np.nan

    # Alinear las columnas del input_df con las que se utilizaron para entrenar el modelo
    expected_columns = df_processed.drop(columns=["prices_per_night"]).columns

    # Agregar columnas faltantes si es necesario
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_columns]  # Asegurar el orden correcto

    # Imputar valores faltantes
    imputed_values = knn_imputer.transform(input_df)
    for i, col in enumerate(columns_to_impute):
        input_df[col] = imputed_values[0][i]

    # Cargar el escalador
    x_scaler, y_scaler = load_scalers()

    # Escalar las características
    features_scaled = x_scaler.transform(input_df)

    # Realizar la predicción
    predicted_price_scaled = model.predict(features_scaled)[0]
    predicted_price = y_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    # Mostrar la predicción
    st.write(f"**The predicted price for this property is: €{predicted_price:.2f}**.")

    st.subheader("Model Evaluation Metrics")
    metrics_df = pd.read_csv("data/lightgbm_metrics.csv")

    st.dataframe(metrics_df)

    with st.expander("Feature Importance Chart"):
    # Llamar la visualizacion en visualizations.py
        show_feature_importance()
    
        st.write("""
        - **Ratings**: Properties with higher ratings are likely to command premium prices, reflecting customer satisfaction.
        - **Number of Reviews**: A higher number of reviews often indicates popularity and demand.
        - **Cleaning Fee**: The additional cleaning fee impacts the total price, highlighting its importance in determining overall cost.
        - **Kitchen and Dining Amenities**: These features contribute significantly to pricing, as they add value for longer stays or family-oriented accommodations.
        - **Exterior Features**: Outdoor spaces or aesthetics enhance property appeal and justify higher prices.
        These factors align with the intuition that customer feedback, service fees, and amenities are key determinants of property pricing in the Airbnb market.
    """)
    

def show_neural_network_price_prediction(df_processed):
    st.header("💸 Price Prediction Model (Neural Networks)")

    models = load_models()
    model = models["neural_network"]

    st.subheader("Please enter the following details about the property:")

    max_guests = st.sidebar.selectbox("Maximum Guests", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
    bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    beds = st.sidebar.selectbox("Number of Beds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5], index=1)
    cleaning_fee = st.sidebar.slider("Cleaning Fee (€)", min_value=0, value=50)

    user_input = pd.DataFrame({
    "maximum_guests": [max_guests],
    "dormitorios": [bedrooms],
    "camas": [beds],
    "baños": [bathrooms],
    "cleaning_fee": [cleaning_fee]
    })

    all_features = df_processed.drop(columns=["prices_per_night"]).columns
    for col in all_features:
        if col not in user_input.columns:
            mean_value = df_processed[col].mean() if col in df_processed.columns else 0
            user_input[col] = mean_value

    user_input = user_input[all_features]        

    x_scaler, y_scaler = load_scalers()
    input_scaled = x_scaler.transform(user_input)

    predicted_price_scaled = model.predict(input_scaled)
    predicted_price = y_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

    st.write(f"**The predicted price for this property is: €{predicted_price:.2f}**.")  

    st.subheader("Model Evaluation Metrics")
    metrics_df = pd.read_csv("data/simple_nn_metrics.csv")
    st.dataframe(metrics_df)

    st.subheader("Model Training and Validation Loss")
    train_val_loss_fig = joblib.load("images/analysis/train_val_loss.pkl")
    st.plotly_chart(train_val_loss_fig)

    st.subheader("Real vs Predicted Prices")
    real_vs_pred_fig = joblib.load("images/analysis/real_vs_pred.pkl")
    st.plotly_chart(real_vs_pred_fig)


def show_recommender_and_nlp(df_sentiment):
    """
    Muestra el Sistema de Recomendaciones y Análisis de Sentimientos en la misma página.
    """
    st.header("🏘️ Recommender System + Sentiment Analysis")
    st.write("Enter details about an Airbnb to get sentiment analysis and similar recommendations.")

    models = load_models()
    nn_model = models["recommendation"]

    st.subheader("Enter the Airbnb features:")
    max_guests = st.sidebar.selectbox("Maximum Guests", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
    bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    beds = st.sidebar.selectbox("Number of Beds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
    bathrooms = st.sidebar.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5], index=1)
    cleaning_fee = st.sidebar.slider("Cleaning Fee (€)", min_value=0, value=50)
    prices_per_night = st.sidebar.slider("Prices per night (€)", min_value=0, max_value=450, value = 120)

    user_input = pd.DataFrame({
        "maximum_guests": [max_guests],
        "bedrooms": [bedrooms],
        "beds": [beds],
        "bathrooms": [bathrooms],
        "cleaning_fee": [cleaning_fee],
        "prices_per_night" : [prices_per_night]
    })

    # Asegurarse de que la entrada del usuario coincida con las características esperadas por el modelo
    all_features = df_sentiment.drop(columns=['url',  'cantidad_comentarios', 'polaridad_media', 'subjetividad_media', 
                                              'palabras_mas_usadas', 'sentimiento','ratings','check_in_hour', 'check_out_hour','total_hours_checkin',
                                              'log_num_reviews','aparcamiento e instalaciones', 'baño', 'calefacción y refrigeración', 'características de la ubicación', 
                                              'cocina y comedor','dormitorio y lavandería', 'entretenimiento', 'exterior','internet y oficina', 'para familias', 
                                              'privacidad y seguridad','seguridad en el hogar', 'servicios', 'habitacion','alojamiento entero']).columns #'prices_per_night',
    
    # Rellenar las columnas faltantes con sus valores medios
    for col in all_features:
        if col not in user_input.columns:
            mean_value = df_sentiment[col].mean() if col in df_sentiment.columns else 0
            user_input[col] = mean_value

    user_input = user_input[all_features]  # Asegurar el orden de las características

    # Botón recomendaciones
    if st.button("Find Similar Airbnbs"):
        recommendations = recommend_airbnbs(user_input, df_sentiment, nn_model)

        st.subheader("Recommended Airbnbs:")
        for i, row in recommendations.iterrows():
            st.write(f"**Recommendation {i + 1}:**")
            display_airbnb_info(row)


def recommend_airbnbs(user_input, df, model):
    """
    Obtener recomendaciones basadas en la entrada del usuario y el modelo entrenado de NearestNeighbors.
    """
    # Usar el modelo para encontrar los vecinos más cercanos
    distances, indices = model.kneighbors(user_input.values)

    # # Recuperar los Airbnbs más similares
    recommended_airbnbs = df.iloc[indices[0]]

    return recommended_airbnbs


def display_airbnb_info(airbnb):
    """
    Muestra la información detallada sobre un Airbnb.
    """
    st.write(f"- **URL**: [View Airbnb]({airbnb['url']})")
    st.write(f"- **Price**: {airbnb['prices_per_night']} €")
    st.write(f"- **Average Polarity**: {airbnb['polaridad_media']}")
    st.write(f"- **Average Subjectivity**: {airbnb['subjetividad_media']}")
    st.write(f"- **Most Used Words**: {airbnb['palabras_mas_usadas']}")
    st.write(f"- **Sentiment**: {airbnb['sentimiento']}")

    
def show_model_explanation(model_choice):
    st.subheader("Model Explanation")

    if model_choice == "Price Prediction - LightGBM":
        st.write("""
        The LightGBM model is a gradient boosting algorithm that predicts property prices based on features such as the number of bedrooms, cleaning fee, and other relevant characteristics.
        This model was trained using a dataset of Airbnb listings, where we carefully tuned its hyperparameters to achieve optimal predictive performance.
        """)
        st.write("""
        To ensure we used the best model for price prediction, we tested several algorithms, including Linear Regression, Random Forest, Gradient Boosting, XGBoost, and MLP Regressor. 
        After evaluating their performance based on their metrics, we found LightGBM to deliver the best balance between accuracy and efficiency.
        """)
        st.markdown("### Comparison of Model Results")
        resultados_modelos = pd.read_pickle("models/resultados_modelos.pkl")
        st.dataframe(resultados_modelos)        

    elif model_choice == "Price Prediction - Neural Networks":
        st.write("""
            The neural network model is based on a deep learning approach, where multiple layers of neurons are trained to predict property prices.
            The network is designed to learn from complex patterns in the data, including interactions between features such as the number of bedrooms, cleaning fee, and other aspects.
        """)
        
    elif model_choice == "Recommender + NLP Sentiment Analysis":
        st.write("""
            The recommendation system suggests Airbnb properties based on user preferences, such as the number of guests, bedrooms, and other attributes.
            The sentiment analysis part analyzes the reviews of the properties to determine whether guests had positive or negative experiences.
            This can help users make informed decisions when selecting a property.
        """)
        
        st.write("Sentiment analysis uses NLP techniques to analyze review text and classify sentiment.")

# Función principal de visualización en la página
def show():
    # Recuperando os dados processados
    df_processed = st.session_state.df_processed
    df_sentiment = st.session_state.df_sentiment

    # Barra lateral para elegir el modelo
    model_choice = st.sidebar.selectbox(
        "Choose the analysis model:",
        ["Price Prediction - LightGBM", "Price Prediction - Neural Networks", "Recommender + NLP Sentiment Analysis"]
    )

    # Mostrar la explicación del modelo seleccionado
    show_model_explanation(model_choice)

    # Mostrar los resultados según la elección del modelo
    if model_choice == "Price Prediction - LightGBM":
        show_price_prediction(df_processed)
    elif model_choice == "Price Prediction - Neural Networks":
        show_neural_network_price_prediction(df_processed)
    elif model_choice == "Recommender + NLP Sentiment Analysis":
        show_recommender_and_nlp(df_sentiment)