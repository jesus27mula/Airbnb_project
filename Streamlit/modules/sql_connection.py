import os
import streamlit as st
from dotenv import load_dotenv

from utils import load_csv, get_db_connection


# Cargar variables de entorno desde .env (si existe)
load_dotenv()

def connect_to_database():
    """
    Conecta al MySQL usando variables de entorno.
    Retorna una conexión o lanza RuntimeError si no se puede conectar.
    """
    try:
        conn = get_db_connection()
        if conn:
            return conn
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ No se pudo conectar a la base de datos: {e}")
        return None
    


def fetch_data(query, fallback_csv=None):
    """
    Ejecuta query y devuelve resultados con fallback a CSV si está disponible.
    
    Args:
        query: Consulta SQL a ejecutar
        fallback_csv: Nombre del archivo CSV como respaldo (ej: 'df_servicios_final_cleaned.csv')
    
    Returns:
        Lista de diccionarios con los resultados o lista vacía si falla
    """
    conn = connect_to_database()
    
    if conn:
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)  # Devuelve diccionarios
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            st.warning(f"⚠️ Error en consulta SQL: {e}")
            # Intentar fallback a CSV si se especificó
            if fallback_csv:
                return fetch_from_csv(fallback_csv, query)
            return []
        finally:
            if cursor:
                cursor.close()
            conn.close()
    else:
        # Si no hay conexión, usar CSV si está disponible
        if fallback_csv:
            st.info("Usando datos desde archivo CSV (modo sin conexión a DB)")
            return fetch_from_csv(fallback_csv, query)
        return []
    
def fetch_from_csv(csv_filename, query):
    """
    Obtiene datos desde un archivo CSV como respaldo.
    Intenta simular la lógica de la consulta SQL.
    
    Args:
        csv_filename: Nombre del archivo CSV en carpeta data/
        query: Consulta SQL original (para lógica de procesamiento)
    
    Returns:
        Lista de diccionarios con los resultados simulados
    """
    try:
        # Cargar el CSV usando utils
        df = load_csv(csv_filename)
        
        if df.empty:
            st.error(f"❌ Archivo CSV de respaldo vacío o no encontrado: {csv_filename}")
            return []
        
        # Lógica específica para la consulta de top 10 servicios
        if "Services_" in query and "service" in query.lower():
            if 'service' in df.columns:
                # Simular la consulta de top 10 servicios
                top_10 = df['service'].value_counts().head(10).reset_index()
                top_10.columns = ['service', 'count']
                return top_10.to_dict('records')
        
        # Lógica genérica: devolver primeras filas
        st.info(f"Mostrando primeros registros de {csv_filename}")
        return df.head(10).to_dict('records')
        
    except Exception as e:
        st.error(f"Error procesando CSV de respaldo: {e}")
        return []

def fetch_top_10_services():
    """
    Obtiene los 10 servicios más ofrecidos en Airbnb.
    Intenta desde MySQL, si falla usa CSV como respaldo.
    
    Returns:
        Lista de diccionarios con 'service' y 'count'
    """
    query = """
        SELECT Services_.service, COUNT(*) as count
        FROM Services_
        JOIN Services_Hosting ON Services_.service_id = Services_Hosting.service_id
        GROUP BY Services_.service
        ORDER BY count DESC
        LIMIT 10;
    """
    
    # Intentar MySQL con fallback a CSV específico
    return fetch_data(query, fallback_csv='df_servicios_final_cleaned.csv')


# Función para ejecutar consultas personalizadas
def execute_custom_query(query, params=None):
    """
    Ejecuta una consulta personalizada con parámetros.
    
    Args:
        query: Consulta SQL con placeholders %s
        params: Tupla con parámetros para la consulta
    
    Returns:
        Lista de resultados o lista vacía
    """
    conn = connect_to_database()
    
    if not conn:
        st.warning("No hay conexión a la base de datos para consulta personalizada")
        return []
    
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        result = cursor.fetchall()
        return result
        
    except Exception as e:
        st.error(f"❌ Error en consulta personalizada: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        conn.close()
        
        
# Para uso directo en debugging
if __name__ == "__main__":
    import pandas as pd
    import streamlit as st
    
    st.set_page_config(layout="wide")
    
    # Probar conexión
    if test_connection():
        st.success("✅ Conexión a MySQL exitosa")
        
        # Mostrar información de la base de datos
        info = get_database_info()
        if 'error' not in info:
            st.write("### Información de la base de datos:")
            st.write(f"- **Versión MySQL:** {info.get('version', 'N/A')}")
            st.write(f"- **Base de datos:** {info.get('database', 'N/A')}")
            st.write(f"- **Tablas disponibles:** {len(info.get('tables', []))}")
            
            if info.get('tables'):
                st.write("Lista de tablas:")
                for table in info['tables']:
                    st.write(f"  - {table}")
        else:
            st.error(f"Error: {info['error']}")
    else:
        st.error("❌ No se pudo conectar a MySQL")
        st.info("📊 Modo: Usando archivos CSV locales")
    
    # Probar consulta de servicios
    st.write("### Probando consulta de servicios:")
    services = fetch_top_10_services()
    
    if services:
        st.success(f"✅ Se obtuvieron {len(services)} servicios")
        df_services = pd.DataFrame(services)
        st.dataframe(df_services)
        
        # Mostrar gráfico simple
        st.bar_chart(df_services.set_index('service')['count'])
    else:
        st.warning("⚠️ No se obtuvieron servicios")
