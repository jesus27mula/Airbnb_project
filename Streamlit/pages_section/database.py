import streamlit as st

def show():
    st.title("🛠️ Database Architecture")

    st.image("images/sqlschema.png", caption="Database Architecture Diagram", width=600)

    st.header("Tables and Columns Overview")

    st.write("""
    This is the database architecture for the project, which consists of several tables storing data about Airbnb listings, 
    their features, and sentiment analysis results.
    
    Below is a description of each table and its columns:
    """)

    st.subheader("1. Hosting Table")
    st.write("""
    This table contains basic information about the Airbnb listing, including the title, type of property, and the host's name.
    
    - **record_id**: Unique identifier for each listing.
    - **titles**: The title or name of the listing.
    - **property_types**: Type of property (e.g., apartment, house).
    - **host_name**: Name of the host.
    """)

    st.subheader("2. Description Table")
    st.write("""
    This table contains more detailed information about the listing, such as pricing, check-in/check-out times, and the number of guests.
    
    - **id**: References the unique record_id from the Hosting table.
    - **prices_per_night**: Price per night for the listing.
    - **check_in_hour**: The check-in time.
    - **check_out_hour**: The check-out time.
    - **total_hours_checkin**: Duration of check-in.
    - **cleaning_fee**: Cleaning fee for the listing.
    - **maximum_guests**: Maximum number of guests allowed.
    - **camas**: Number of beds.
    - **baños**: Number of bathrooms.
    - **dormitorios**: Number of bedrooms.
    """)

    st.subheader("3. Ratings Table")
    st.write("""
    This table contains sentiment-related data about the listing, such as ratings, sentiment scores, and most used words.
    
    - **record_id**: References the unique record_id from the Hosting table.
    - **ratings**: Average rating of the listing.
    - **num_reviews**: Total number of reviews for the listing.
    - **polaridad_media**: Average polarity score from sentiment analysis.
    - **subjetividad_media**: Average subjectivity score from sentiment analysis.
    - **palabras_mas_usadas**: Most frequent words used in reviews.
    - **sentimiento**: Overall sentiment of the reviews (positive, negative, neutral).
    """)

    st.subheader("4. Services Table")
    st.write("""
    This table contains a list of all available services that a listing can have, such as Wi-Fi, parking, etc.
    
    - **service_id**: Unique identifier for each service.
    - **service**: Name of the service (e.g., Wi-Fi, pool, etc.).
    """)

    st.subheader("5. Services_Hosting Table")
    st.write("""
    This table links services to specific listings.
    
    - **service_id**: References the unique service_id from the Services table.
    - **record_id**: References the unique record_id from the Hosting table.
    """)

    st.subheader("6. Category Table")
    st.write("""
    This table defines the categories of services (e.g., basic, premium).
    
    - **category_id**: Unique identifier for each category.
    - **category**: Name of the category
    """)

    st.subheader("7. Category_Services Table")
    st.write("""
    This table links services to their respective categories.
    
    - **service_id**: References the unique service_id from the Services table.
    - **category_id**: References the unique category_id from the Category table.
    """)