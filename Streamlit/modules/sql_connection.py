import mysql.connector

def connect_to_database():
    """
    Conecta ao banco de dados MySQL.
    """
    connection = mysql.connector.connect(
        host="localhost",
        user="root",       
        password="password",
        database="AIRBNB" 
    )
    return connection


def fetch_data(query):

    connection = connect_to_database()
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result


def fetch_top_10_services():
    
    query = """
        SELECT Services_.service, COUNT(*) as count
        FROM Services_
        JOIN Services_Hosting ON Services_.service_id = Services_Hosting.service_id
        GROUP BY Services_.service
        ORDER BY count DESC
        LIMIT 10;
    """
    return fetch_data(query)

