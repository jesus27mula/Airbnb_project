import streamlit as st

def show():
    st.title("About Us")
    
    st.write("""
    Welcome to our project! This application was developed as part of our final project for the Data Science course. 
    Throughout this journey, we’ve applied our skills in data analysis and machine learning to create a powerful and interactive tool.
    We invite you to explore our work and connect with us below. We are always open to collaboration and new opportunities in the tech
    and data science field!

    Feel free to reach out to us through our profiles.
    """)

    col1, col2 = st.columns(2)

    # Perfil Diogo
    with col1:
        st.subheader("Diogo Bernardo")
        st.markdown(
            """
            <p>
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20" style="vertical-align: middle; margin-right: 10px;">
                <a href="https://github.com/DiogoBernardoPT" target="_blank">GitHub Profile</a>
            </p>
            <p>
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" style="vertical-align: middle; margin-right: 10px;">
                <a href="https://www.linkedin.com/in/diogogalhanas" target="_blank">LinkedIn Profile</a>
            </p>
            """,
            unsafe_allow_html=True,
        )

    # Perfil Jesus
    with col2:
        st.subheader("Jesus Domènech")
        st.markdown(
            """
            <p>
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20" style="vertical-align: middle; margin-right: 10px;">
                <a href="https://github.com/jesus27mula" target="_blank">GitHub Profile</a>
            </p>
            <p>
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" style="vertical-align: middle; margin-right: 10px;">
                <a href="https://www.linkedin.com/in/jesus-maria-mulà-domènech" target="_blank">LinkedIn Profile</a>
            </p>
            """,
            unsafe_allow_html=True,
        )
