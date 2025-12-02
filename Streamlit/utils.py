import os
import sys
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import streamlit as st

# ============================================
# CONFIGURACIÓN DE RUTAS 
# ============================================


def get_project_root():
    
    #Devuelve la ruta absoluta de la carpeta Streamlit
    return os.path.dirname(os.path.abspath(__file__))


def get_data_path(filename=None):
    #Devuelve la ruta absoluta de la carpeta data
    
    data_dir = os.path.join(get_project_root(), 'data')
    
    if filename:
        filepath = os.path.join(data_dir, filename)
        return filepath
    return data_dir   


def get_models_path(filename=None):
    #Devuelve la ruta absoluta de la carpeta models
    
    models_dir = os.path.join(get_project_root(), 'models')
    
    if filename:
        return os.path.join(models_dir, filename)
    return models_dir


def get_images_path(filename=None, subfolder=None):
    
    #Devuelve la ruta absoluta de la carpeta images
    
    images_dir = os.path.join(get_project_root(), 'images')
    
    if subfolder:
        images_dir = os.path.join(images_dir, subfolder)
        
    if filename:
        if '/' in filename or '\\' in filename:
            return os.path.join(get_project_root(), 'images', filename)
        else:
            # Primero buscar en la ruta especificada (con subfolder si hay)
            filepath = os.path.join(images_dir, filename)
            
            # Si no existe, buscar recursivamente
            if not os.path.exists(filepath):
                for root, dirs, files in os.walk(os.path.join(get_project_root(), 'images')):
                    if filename in files:
                        return os.path.join(root, filename)
            
            return filepath
        
    return images_dir

def get_modules_path(filename=None):
    #Devuelve la ruta absoluta de la carpeta modules
    modules_dir = os.path.join(get_project_root(), 'modules')
    
    if filename:
        return os.path.join(modules_dir, filename)
    return modules_dir

def get_pages_section_path(filename=None):
    #Devuelve la ruta absoluta de la carpeta pages_section
    pages_dir = os.path.join(get_project_root(), 'pages_section')
    
    if filename:
        return os.path.join(pages_dir, filename)
    return pages_dir


# ============================================
# FUNCIONES DE CARGA DE DATOS
# ============================================

def load_csv(filename, usecols=None, nrows=None):
    """
    Carga un archivo CSV desde la carpeta data/
    
    Args:
        filename: Nombre del archivo CSV
        usecols: Columnas específicas a cargar
        nrows: Número máximo de filas a cargar
    
    Returns:
        DataFrame con los datos
    """
    try:
        filepath = get_data_path(filename)
        
        df = pd.read_csv(filepath, usecols=usecols, nrows=nrows)
        return df
    
    except Exception as e:
        st.error(f"Error cargando el archivo {filename}: {e}")
    return pd.DataFrame()


def load_model(model_filename):
    """
    Carga un modelo desde la carpeta models/
    
    Args:
        model_filename: Nombre del archivo del modelo (.pkl, .joblib, etc.)
    
    Returns:
        Modelo cargado o None
    """
    
    try:
        import joblib
        
        model_path = get_models_path(model_filename)
        
        # Cargar el modelo
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando modelo {model_filename}: {(e)}")
        return None
    
def load_image(image_filename, subfolder='analysis'):
    """
    Carga una imagen desde la carpeta images/
    
    Args:
        image_filename: Nombre del archivo de imagen
        subfolder: Subcarpeta dentro de images/ (default: 'analysis')
    
    Returns:
        Imagen cargada o None
    """
    try:
        from PIL import Image
        
        # Primero intentar con subfolder
        image_path = get_images_path(image_filename, subfolder=subfolder)
        
        if not os.path.exists(image_path):
            # Si no existe, buscar sin subfolder
            image_path = get_images_path(image_filename)
            if not os.path.exists(image_path):
                # Buscar recursivamente en toda la carpeta images
                images_dir = get_images_path()
                found = False
                if os.path.exists(images_dir):
                    for root, dirs, files in os.walk(images_dir):
                        if image_filename in files:
                            image_path = os.path.join(root, image_filename)
                            found = True
                            st.info(f"🔍 Imagen encontrada en: {os.path.relpath(image_path, images_dir)}")
                            break
                        
                if not found:
                    st.error(f"Imagen no encontrada: {image_filename}")
                    return None
                    
        # Cargar la imagen
        img = Image.open(image_path)
        return img
    except Exception as e:
        st.error(f"Error cargando imagen {image_filename}: {str(e)}")
        return None
    
def load_pickle(pickle_filename, subfolder='analysis'):
    try:
        import joblib
        
        # Obtener la ruta al archivo .pkl
        pickle_path = get_images_path(pickle_filename, subfolder=subfolder)
        
        if not os.path.exists(pickle_path):
            st.error(f"Archivo .pkl no encontrado: {pickle_path}")
            
            # Buscar recursivamente
            images_dir = get_images_path()
            found = False
            if os.path.exists(images_dir):
                for root, dirs, files in os.walk(images_dir):
                    if pickle_filename in files:
                        pickle_path = os.path.join(root, pickle_filename)
                        found = True
                        break
            
            if not found:
                return None
        
        # Cargar el archivo .pkl
        obj = joblib.load(pickle_path)
        return obj
        
    except ImportError:
        st.error("joblib no está instalado. Ejecuta: pip install joblib")
        return None
    except Exception as e:
        st.error(f"Error cargando {pickle_filename}: {str(e)}")
        return None
    
def load_analysis_pickle(pickle_filename):
    """
    Atajo para cargar archivos .pkl de la carpeta analysis/
    """
    return load_pickle(pickle_filename, subfolder='analysis')

# ============================================
# FUNCIONES 
# ============================================

def load_analysis_image(image_filename):
    """
    Atajo para cargar imágenes de la carpeta analysis/
    """
    return load_image(image_filename, subfolder='analysis')

# ============================================
# CONEXIÓN A BASE DE DATOS
# ============================================

def get_db_connection():
    """
    Conecta a MySQL usando credenciales del archivo .env.local
    """
    try:
        # Cargar variables de entorno
        env_file = os.path.join(get_project_root(), '.env.local')
        if not os.path.exists(env_file):
            return None
        
        load_dotenv(env_file)
        
            
        required_vars = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', 3306),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
        }
        
        # Verificar que ninguna variable requerida sea None o vacía
        for key, value in required_vars.items():
            if not value:
                print(f"⚠️ Missing required environment variable: DB_{key.upper()}")
                return None
        
        config = {
            'host': required_vars['host'],
            'port': int(required_vars['port']),
            'database': required_vars['database'],
            'user': required_vars['user'],
            'password': required_vars['password'],
            'auth_plugin': 'mysql_native_password'
        }
            
        conn = mysql.connector.connect(**config)
        return conn
    
    except mysql.connector.Error as e:
        # Manejar errores específicos de MySQL
        error_msg = str(e)
        if "Access denied" in error_msg:
            print("⚠️ Access denied to MySQL database")
        elif "Unknown database" in error_msg:
            print("⚠️ Database does not exist")
        else:
            print(f"⚠️ MySQL connection error: {e}")
        return None
    except Exception as e:
        # Error genérico
        print(f"⚠️ Error connecting to database: {e}")
        return None

# ============================================
# FUNCIONES DE INICIALIZACIÓN
# ============================================

def load_all_dataframes():
    required_files = {
        'df': 'df_final_cleaned.csv',
        'df_processed': 'df_processed_to_ML.csv',
        'df_sentiment': 'df_rec_st.csv'
    }
    
    for key, filename in required_files.items():
        if key not in st.session_state:
            df = load_csv(filename)
            st.session_state[key] = df
            
def init_session_state():
    """
    Inicializa todas las variables de session_state
    """
    if 'initialized' not in st.session_state:
        load_all_dataframes()
        st.session_state.initialized = True
        

# ============================================
# PARA IMPORTAR EN OTROS MÓDULOS
# ============================================

def add_project_to_path():
    """
    Añade el proyecto al path de Python para imports
    """
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        return True
    return False