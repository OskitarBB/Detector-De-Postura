import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import glob
from collections import deque
import sys
import threading
import time
from PyQt6.QtCore import QObject, pyqtSignal

# Importamos winsound solo si estamos en Windows
if sys.platform.startswith('win'):
    import winsound

# --- CONFIGURACIÓN DE ARCHIVOS Y CARPETAS ---
PERFILES_DIR = 'PERFILES'
TRAINING_FILE_PATTERN = 'entrenamiento_*.json'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Definición de la instancia pose 
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONTROL ASÍNCRONO DE ALARMA ---

# Globales para el control del hilo de la alarma
current_alarm_thread = None 
alarm_stop_event = threading.Event()

def run_alarm_task(duration_ms):
    """Tarea que se ejecuta en un hilo, reproduce el beep, y se detiene si se dispara el evento."""
    global current_alarm_thread
    
    alarm_stop_event.clear()

    start_time = time.time()
    end_time = start_time + (duration_ms / 1000.0)
    
    # Bucle para mantener el sonido activo (o simularlo) y verificar si debe parar
    while time.time() < end_time and not alarm_stop_event.is_set():
        if sys.platform.startswith('win'):
            # Beep corto de 100ms para simular un tono continuo de 5s
            winsound.Beep(400, 100) 
        else:
            print("[ALERTA AUDITIVA] BEEP! (simulando...)")
        
        time.sleep(0.1) 
    
    current_alarm_thread = None # La alarma ha terminado su ciclo o fue detenida

def disparar_alarma_interruptible():
    """Llama a la alarma en un hilo y guarda la referencia, SOLO si no está sonando."""
    global current_alarm_thread
    
    # Solo dispara si NO hay un hilo de alarma activo
    if current_alarm_thread is None or not current_alarm_thread.is_alive():
        current_alarm_thread = threading.Thread(target=run_alarm_task, args=(5000,)) # 5 segundos
        current_alarm_thread.start()

def detener_alarma():
    """Dispara el evento para que el hilo de la alarma detenga su ciclo."""
    global current_alarm_thread
    if current_alarm_thread and current_alarm_thread.is_alive():
        alarm_stop_event.set()
        print("[ALERTA] Alarma interrumpida.")

# --- CLASE: FILTRO DE PREDICCIÓN (Usado en run_detector.py) ---

class PredictionFilter:
    """Clase para determinar la predicción dominante en una ventana de frames."""
    def __init__(self, window_size=15):
        self.window = deque(maxlen=window_size)
    
    def add_prediction(self, prediction):
        self.window.append(prediction)
    
    def get_dominant_prediction(self):
        """Devuelve la clase ('PERFECTO', 'MALO', etc.) más común en la ventana."""
        if not self.window: return 'Buscando'
        counts = {}
        for p in self.window:
            counts[p] = counts.get(p, 0) + 1
        
        return max(counts, key=counts.get)

# --- FUNCIONES DE PERSISTENCIA Y RUTAS ---

def obtener_ruta_perfil(nombre_perfil):
    """Devuelve la ruta de la carpeta de un perfil y se asegura de que exista."""
    ruta_perfil = os.path.join(PERFILES_DIR, nombre_perfil)
    os.makedirs(ruta_perfil, exist_ok=True)
    return ruta_perfil

def obtener_nombres_de_perfiles():
    """Devuelve una lista de todos los perfiles existentes (nombres de subcarpetas)."""
    if not os.path.exists(PERFILES_DIR):
        return []
    
    return [d for d in os.listdir(PERFILES_DIR) if os.path.isdir(os.path.join(PERFILES_DIR, d))]

def guardar_entrenamiento_bruto(nombre_perfil, data_angulos_sesion):
    """Guarda la sesión de entrenamiento actual en un archivo JSON numerado."""
    ruta_perfil = obtener_ruta_perfil(nombre_perfil)
    
    archivos_existentes = glob.glob(os.path.join(ruta_perfil, TRAINING_FILE_PATTERN.replace('*', '[0-9]*')))
    
    nueva_version = len(archivos_existentes) + 1
    nombre_archivo = f"entrenamiento_{nueva_version:03d}.json"
    
    ruta_completa = os.path.join(ruta_perfil, nombre_archivo)
    
    with open(ruta_completa, 'w') as f:
        json.dump(data_angulos_sesion, f, indent=4)
        
    print(f"[INFO] Sesión de entrenamiento guardada como: {nombre_archivo}")

def cargar_datos_brutos_para_recalculo(nombre_perfil):
    """Carga y consolida TODOS los datos brutos de entrenamiento de una carpeta de perfil."""
    ruta_perfil = obtener_ruta_perfil(nombre_perfil)
    
    datos_consolidados = {
        'PERFECTO': [], 
        'MALO': []
    }
    
    archivos_entrenamiento = glob.glob(os.path.join(ruta_perfil, TRAINING_FILE_PATTERN))
    
    for archivo in archivos_entrenamiento:
        try:
            with open(archivo, 'r') as f:
                data_sesion = json.load(f)
                
                if 'PERFECTO' in data_sesion and 'MALO' in data_sesion:
                    datos_consolidados['PERFECTO'].extend(data_sesion['PERFECTO'])
                    datos_consolidados['MALO'].extend(data_sesion['MALO'])
                    
        except Exception as e:
            print(f"[ERROR] Error al leer archivo {archivo}: {e}")
            
    return datos_consolidados

# --- EXTRACCIÓN DE FEATURES ML ---

def extraer_features(landmarks):
    """
    Extrae las 33 coordenadas X, Y, Z normalizadas de los landmarks y las concatena 
    en un vector de 99 elementos (nuestro vector de características para ML).
    """
    
    try:
        # Crea un array numpy de 99 elementos (33 * 3)
        features = np.array([[res.x, res.y, res.z] for res in landmarks]).flatten().tolist()
        return features
    except:
        # Devuelve un vector de ceros si la detección falla
        return [0.0] * 99

def clasificar_postura(prediccion_ml):
    """Clasifica la postura basada en el resultado del modelo ML."""
    
    if prediccion_ml == 'PERFECTO':
        return "Postura Correcta", (0, 255, 0)
    elif prediccion_ml == 'ACEPTABLE':
        return "Zona de Alerta", (0, 165, 255)
    elif prediccion_ml == 'MALO':
        return "Postura Incorrecta", (255, 0, 0) # Cambiado a (255, 0, 0) para rojo puro
    else:
        return "Buscando...", (255, 255, 255)