import streamlit as st

st.set_page_config(page_title='Airbnb Data Dashboard', page_icon='🏠', layout='wide')

def show():

    st.title('Welcome to the Airbnb Data Analysis Dashboard')

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('''
        **Welcome to the intersection of data, innovation, and hospitality!**  
        This project explores Airbnb listings in Barcelona, providing actionable insights through a comprehensive and interactive dashboard.  
        From understanding market trends to predicting prices, this tool combines advanced analytics with user-friendly design.

        ### Project Overview  
        This project follows a well-structured workflow to deliver meaningful insights:
        
        - **Phase 1: Data Extraction**  
        We gathered detailed data from Airbnb listings in Barcelona, using web scraping techniques to ensure a comprehensive dataset.
        
        - **Phase 2: Data Cleaning**  
        Through meticulous preprocessing, we cleaned the data to handle missing values, remove duplicates, and standardize key fields using **pandas**. 
        This phase ensured the dataset was ready for robust analysis and to prepare for machine learning.
        
        - **Phase 3: Exploratory Data Analysis (EDA)**  
        Using **Seaborn**, **Matplotlib**, and **Plotly**, we visualized key trends in pricing, ratings, and guest preferences. Correlation analysis, boxplots, and histograms revealed hidden patterns.  
        These insights are presented interactively on the **Data Analysis** page of this dashboard.  
        You should try it!        
        
        - **Phase 4: SQL Integration**  
        To optimize performance and manage large datasets efficiently, we transitioned from CSV files to a relational **MySQL** database, enabling seamless querying and scalability.
        
        - **Phase 5: Machine Learning Preprocessing**  
        Features were engineered, scaled, and prepared to build predictive models, setting the foundation for accurate and reliable results.
        
        - **Phase 6: Machine Learning**  
        Advanced regression models, including **LightGBM** and **neural networks**, were implemented to predict property prices and uncover the driving factors behind them.
        
        - **Phase 7: Streamlit Dashboard**  
        This dashboard was developed to deliver insights interactively, providing visualizations, recommendations, and a platform for future exploration.
        
        ### Your Experience  
        Use this dashboard to explore pricing trends, filter listings based on key features, and gain a deeper understanding of the Airbnb market in Barcelona. Whether you're a data enthusiast, a business professional, or simply curious about the Airbnb business, this tool is designed to make complex data accessible and actionable.  

        Ready to start your journey? 🏠✨
        ''')

    with col2:
        st.image('images/airbnb_1stpage.png', use_column_width=True)
        st.image('images/airbnb_1stpage_2.png', use_column_width=True)
        st.image('images/language tools Final project.png', use_column_width=True)