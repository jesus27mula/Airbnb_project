import streamlit as st
import pandas as pd
from modules import visualizations
from modules.sql_connection import fetch_top_10_services


def show(df):
    st.title('**Discover the world of Airbnb accommodations through data!**')
    st.write(''' 
    This dashboard allows you to explore prices, ratings, and property locations in the city of Barcelona. 
    Analyze trends, filter based on your preferences, and uncover the most interesting insights 
    in every corner of the Airbnb market.  
    Ready to start exploring? 🏠✨
    ''')
    st.image('images/captivating_barcelona.png', width=900)

    st.header('Exploratory Data Analysis')
    st.write('''
    In this section, we will explore key insights from the Airbnb dataset. 
    We will analyze trends in pricing, property types, ratings, and reviews 
    to uncover patterns and correlations that can help better understand 
    the dynamics of the Airbnb market in Barcelona.
    ''')

    st.subheader('First Rows of the DataFrame')
    df_display = df.drop(columns=['timestamp', 'record_id', 'titles', 'location', 'host_name'])
    st.dataframe(df_display.head())

    df_filtered = df.copy()

    # Filtros interactivos
    property_type = st.multiselect('Select property type:', options=df['property_types'].unique(), default=None)
    rating_options = ['1-2', '3-4', '4-5']
    selected_rating_range = st.selectbox('Select a rating range:', options=rating_options)
    num_reviews_slider = st.slider(
        'Select minimum number of reviews:',
        min_value=0,
        max_value=int(df['num_reviews'].max()),
        value=0,
        step=1,
        help="Move the slider to select the desired number of reviews."
    )
    price_range = st.slider('Price range per night:', 
                            min_value=int(df['prices_per_night'].min()), 
                            max_value=int(df['prices_per_night'].max()), 
                            value=(int(df['prices_per_night'].min()), int(df['prices_per_night'].max())), 
                            format='€%d')

    apply_filters = st.button('Apply Filters')

    if apply_filters:
        if selected_rating_range == '1-2':
            df_filtered = df_filtered[(df_filtered['ratings'] >= 1) & (df_filtered['ratings'] <= 2)]
        elif selected_rating_range == '3-4':
            df_filtered = df_filtered[(df_filtered['ratings'] > 2) & (df_filtered['ratings'] <= 4)]
        elif selected_rating_range == '4-5':
            df_filtered = df_filtered[(df_filtered['ratings'] > 4) & (df_filtered['ratings'] <= 5)]
        
        if num_reviews_slider > 0:
            df_filtered = df_filtered[df_filtered['num_reviews'] >= num_reviews_slider]
        
        if property_type:
            df_filtered = df_filtered[df_filtered['property_types'].isin(property_type)]
        
        # Aplicar el filtro del precio
        df_filtered = df_filtered[(df_filtered['prices_per_night'] >= price_range[0]) & 
                                  (df_filtered['prices_per_night'] <= price_range[1])]

        st.subheader('Filtered Listings')
        df_filtered_display = df_filtered.drop(columns=['timestamp', 'record_id', 'titles', 'location', 'host_name'])
        st.dataframe(df_filtered_display.head())
    else:
        st.subheader('Listings Without Filters')
        df_display = df.drop(columns=['timestamp', 'record_id', 'titles', 'location', 'host_name'])
        st.dataframe(df_display.head())

    # Visualizaciones iniciales de precio y outliers
    col1, col2 = st.columns(2)
    with col1:
        visualizations.load_and_display_pickle('images/analysis/price_boxplot.pkl')
    with col2:
        visualizations.load_and_display_pickle('images/analysis/price_zscore.pkl')

    with st.expander("Click to see insights from the Z-score analysis"):
        st.write(""" 
    - **Price Distribution with Z-score**: The price analysis shows where most prices are concentrated.
        However, there are some higher prices, between 200 and 239, that exceed the Z-score limit. 
        While these values are technically outliers, they do not necessarily indicate erroneous data but rather higher prices 
        that may correspond to luxury accommodations or those with special features. In these cases, is important to understand
        that not all outliers should be removed. Instead, they should be analyzed in context to uncover insights about unique
        market segments, such as premium offerings or niche properties that cater to specific customer preferences.
        """)

# Visualizaciones EDA lado a lado
    st.subheader('**Data Visualizations**')
    col1, col2 = st.columns(2)

    with col1:
        visualizations.price_rating_distribution(df_filtered)
    with col2:
        visualizations.average_price_by_capacity(df_filtered)

# Anadir hipotesis aqui
    with st.expander("Click to see insights from the graphs above"):
        st.write("""
    - **Price Rating Distribution**: This visualization provides insights into the relationship between property ratings and nightly
        prices across various property types.it becomes evident that higher-rated properties often align with mid-range pricing, 
        but there are exceptions. Some unique property types maintain both high ratings and significantly higher prices.
        This analysis not only identifies key patterns but also underscores how property type and guest ratings influence pricing
        dynamics, offering valuable insights into market positioning and customer preferences.
    - **Average Price by Maximum Guest Capacity**: This visualization highlights the relationship between the maximum guest capacity of a
        property and its average price. As expected, there is a general trend where properties with a higher guest 
        capacity tend to have higher nightly prices. However, it is worth noting that while the trend is generally linear, there 
        are some exceptions. These anomalies might be influenced by other factors, such as property location, luxury features, 
        or unique offerings that make smaller-capacity listings more expensive than expected.        
        """)

    with st.expander("Correlation Map"):
        visualizations.correlation(df_filtered)
        st.write("""
    - **Correlation Insights**: This map highlights relationships between variables, with darker colors 
        indicating stronger correlations. Pay special attention to the correlation between price and guest capacity,
        as well as rooms, beds, baths and cleaning fee. This suggests that larger accommodations or those offering 
        more amenities generally command higher prices. Similarly, the cleaning fee's strong correlation with price 
        highlights how additional services can influence the overall cost.
    """)

    col1, col2 = st.columns(2)
    with col1:
        visualizations.price_property_types(df_filtered)
    with col2:    
        visualizations.price_distribution_histogram(df_filtered)

    with st.expander("Click to see insights from the price analysis"):
        st.write("""
    - **Price by Property Type**: A boxplot analysis reveals that entire accommodations tend to be more expensive than private rooms, 
        as expected. The average price for an entire property is approximately €105 per night, while private rooms average around €45.
        This distinction highlights the premium guests are willing to pay more for exclusive access and additional space.                   
    - **Price Distribution**: A histogram showing the overall distribution of prices across all listings. The majority of listings fall 
        within a mid-range price bracket(30-50€) with a normal distribution. There are noticeable prices on the higher end, which 
        correspond to luxury properties or unique experiences. 
        Understanding the spread of prices provides insights into market trends and potential pricing strategies.
    """)

# Visualizaciones relacionadas a Reviews/Ratings
    col1, col2 = st.columns(2)
    with col1:
        visualizations.rating_distribution(df_filtered)
    with col2:
        visualizations.reviews_rating_distribution(df_filtered)

    visualizations.reviews_price_scatter(df_filtered)

    with st.expander("Click to see insights from the reviews and ratings visualizations"):
        st.write("""
    - **Rating Distribution**: The distribution of ratings for Airbnb listings shows a clear pattern where most properties receive high
        ratings. A significant proportion of listings have ratings between 4.25 and 5, suggesting that guests generally have positive 
        experiences.Additionally, the presence of ratings close to 0 may indicate newly listed properties that have not yet accumulated many reviews. 
    - **Reviews vs Rating**: The scatter plot reveals a positive correlation between the number of reviews, ratings, and the price of 
        properties. As expected, listings with higher ratings and a greater number of reviews tend to have higher prices. This could 
        suggest that more established properties with positive reviews command higher prices due to their proven track record and customer satisfaction.
        This pattern reinforces the idea that higher ratings and reviews can drive up demand and pricing, potentially due to increased 
        visibility and being trust worthy on the platform.
    - **Reviews vs Price**: This illustrates the relationship between the number of reviews and the price per night for Airbnb listings. 
        While there is a general trend showing that properties with a higher number of reviews tend to have higher prices, the data reveals
        some interesting insights. Lower-priced listings can still receive substantial positive attention and many reviews, possibly 
        because they appeal to a larger number of guests looking for good budget options.The rest of the data appears to follow a more 
        typical distribution, where properties with more reviews often maintain higher prices, but there is some randomness in this 
        relationship. This randomness could be due to various factors such as : unique property features, seasonal price fluctuations, or
        differing target audiences.
    """)

    st.header('Top 10 Services Offered on Airbnb')
    visualizations.top_10_services_chart()

    with st.expander("Click to see insights from the services analysis"):
        st.write("""
    - **Top 10 Services**: The most commonly offered services on Airbnb are likely to be those that provide fundamental comfort and 
        convenience for guests, such as kitchen facilities, Wi-Fi, hot water, and basic appliances. Travelers prioritize essential services
        like Wi-Fi for connectivity, kitchen facilities for meal preparation, and hot water and laundry facilities for comfort during 
        their stay, which is expected.
    """)

