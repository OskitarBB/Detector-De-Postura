import cv2
import sys
import warnings
import numpy as np
import time
from collections import deque

# Importaciones de PyQt (CORREGIDO)
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QInputDialog, QMessageBox, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QTextCursor # <-- CORRECCIÓN APLICADA AQUÍ

# Importaciones de la Lógica del Motor
from posture_logic import mp_pose, mp_drawing, pose, obtener_nombres_de_perfiles, extraer_features, guardar_entrenamiento_bruto, obtener_ruta_perfil

# Ocultar warnings de librerías
warnings.filterwarnings("ignore")

# --- CLASE DE REDIRECCIÓN DE CONSOLA ---
class ConsoleRedirect(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass

# --- CONFIGURACIÓN GLOBAL DE CAPTURA ---
DURACION_CAPTURA = 10 # Segundos por postura

# --- CLASE DE LA VENTANA DE ENTRENAMIENTO ---

class PostureTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Redirigir la salida de la consola
        self.console_redirect = ConsoleRedirect()
        self.console_redirect.text_written.connect(self.update_log)
        
        self.setWindowTitle("Detector de Postura - MODO ENTRENAMIENTO (ML)")
        self.setGeometry(100, 100, 1000, 650) 
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        # Estado del entrenamiento
        self.nombre_perfil = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_state = "SELECT_PROFILE"
        
        # Datos de captura de la sesión actual
        self.data_features = {'PERFECTO': [], 'MALO': []}
        self.capture_start_time = 0
        self.CAPTURE_DURATION = DURACION_CAPTURA

        self.setup_ui()
        self.init_camera()
        self.set_state("SELECT_PROFILE") 

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Panel Izquierdo (Control de Entrenamiento y Log)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Título
        title_label = QLabel("MODO ENTRENAMIENTO")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        left_panel.addWidget(title_label)
        left_panel.addSpacing(10)

        # 1. Indicador de Perfil
        self.profile_info = QLabel("Perfil: (No Seleccionado)")
        left_panel.addWidget(self.profile_info)

        # 2. Botón de Selección/Creación
        self.select_button = QPushButton("SELECCIONAR / CREAR PERFIL")
        self.select_button.setStyleSheet("background-color: #007ACC; color: white; padding: 8px; font-weight: bold;")
        self.select_button.clicked.connect(self.show_profile_dialog)
        left_panel.addWidget(self.select_button)
        left_panel.addSpacing(20)

        # 3. Indicador de Etapa
        self.stage_label = QLabel("Etapa Actual: Esperando Perfil...")
        self.stage_label.setFont(QFont("Arial", 14))
        self.stage_label.setStyleSheet("color: #FFD700; border: 1px solid #FFD700;")
        left_panel.addWidget(self.stage_label)

        # 4. Botón de Captura (Acción principal)
        self.capture_button = QPushButton("EMPEZAR")
        self.capture_button.setStyleSheet("background-color: #AAAAAA; color: black; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.handle_capture_click)
        left_panel.addWidget(self.capture_button)
        
        # Indicador de conteo
        self.countdown_label = QLabel("")
        self.countdown_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        left_panel.addWidget(self.countdown_label)
        
        # 5. Área de Log / Consola
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setStyleSheet("background-color: #000000; color: #FFFFFF; border: 1px solid #555;")
        left_panel.addWidget(QLabel("Log del Sistema:"))
        left_panel.addWidget(self.log_widget)
        
        control_widget = QWidget()
        control_widget.setLayout(left_panel)
        control_widget.setFixedWidth(415)
        main_layout.addWidget(control_widget)

        # Panel Derecho (Cámara)
        self.camera_label = QLabel("Cargando Cámara...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; border: 2px solid #555555;")
        self.camera_label.setMinimumSize(600, 400)
        main_layout.addWidget(self.camera_label)
        
    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_label.setText("ERROR: Cámara no disponible.")
        else:
            self.timer.start(30) # ~33 FPS

    def update_log(self, text):
        """Añade texto al widget de log."""
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_widget.setTextCursor(cursor)

    def show_profile_dialog(self):
        perfiles = obtener_nombres_de_perfiles()
        
        items = ["(CREAR NUEVO PERFIL)"] + perfiles
        item, ok = QInputDialog.getItem(self, "Selección de Perfil", "Elija un perfil:", items, 0, False)

        if ok and item:
            if item == "(CREAR NUEVO PERFIL)":
                text, ok_text = QInputDialog.getText(self, 'Nuevo Perfil', 'Ingrese el nombre:')
                if ok_text and text:
                    self.nombre_perfil = text.strip()
            else:
                self.nombre_perfil = item
            
            if self.nombre_perfil:
                self.set_state("READY_PERFECT")
    
    def handle_capture_click(self):
        if self.current_state == "READY_PERFECT":
            self.capture_start_time = time.time()
            self.set_state("CAPTURING_PERFECTO")
            print(f"-> Iniciando captura de PERFECTO...")

        elif self.current_state == "READY_MALO":
            self.capture_start_time = time.time()
            self.set_state("CAPTURING_MALO")
            print(f"-> Iniciando captura de MALO...")


    def set_state(self, new_state):
        self.current_state = new_state
        
        # Resetear estilos y habilitación
        self.capture_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.capture_button.setStyleSheet("background-color: #AAAAAA; color: black; padding: 15px; font-weight: bold; border-radius: 5px;")
        
        
        if new_state == "SELECT_PROFILE":
            self.stage_label.setText("Selecciona un Perfil para comenzar...")
            self.stage_label.setStyleSheet("color: #FFD700; border: 1px solid #FFD700;")
            self.select_button.setEnabled(True)
            self.profile_info.setText("Perfil: (No Seleccionado)")
            self.countdown_label.setText("")

        elif new_state == "READY_PERFECT":
            self.stage_label.setText("Captura los frames para la Postura Correcta")
            self.stage_label.setStyleSheet("color: #00FF00; border: 1px solid #00FF00;")
            self.capture_button.setText("CLIC para EMPEZAR")
            self.capture_button.setStyleSheet("background-color: #00AA00; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
            self.capture_button.setEnabled(True)
            self.profile_info.setText(f"Perfil: {self.nombre_perfil} (Añadiendo datos)")
            self.countdown_label.setText(f"Duración: {self.CAPTURE_DURATION}s")
            
        elif new_state == "CAPTURING_PERFECTO":
            self.stage_label.setText("Capturando frames de Postura Correcta...")
            self.stage_label.setStyleSheet("color: #FFD700; border: 1px solid #FFD700;")
            self.capture_button.setText("CAPTURA ACTIVA...")
            
        elif new_state == "READY_MALO":
            self.stage_label.setText("Etapa Actual: 2/2 (Postura PELIGROSA)")
            self.stage_label.setStyleSheet("color: #FF0000; border: 1px solid #FF0000;")
            self.capture_button.setText("CLIC para EMPEZAR")
            self.capture_button.setStyleSheet("background-color: #FF0000; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
            self.capture_button.setEnabled(True)
            self.countdown_label.setText(f"Duración: {self.CAPTURE_DURATION}s")

        elif new_state == "CAPTURING_MALO":
            self.stage_label.setText("Capturando frames de Postura Incorrecta...")
            self.stage_label.setStyleSheet("color: #FFD700; border: 1px solid #FFD700;")
            self.capture_button.setText("CAPTURA ACTIVA...")

        elif new_state == "FINISHED":
            self.stage_label.setText("ENTRENAMIENTO FINALIZADO. Guardando...")
            self.stage_label.setStyleSheet("color: #00FF00; border: 1px solid #00FF00;")
            self.capture_button.setText("Entrenamiento Finalizado")
            self.capture_button.setEnabled(False)
            self.countdown_label.setText("")


    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            return

        img = cv2.flip(img, 1) 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        # Dibuja landmarks de OpenCV
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Lógica de Captura
            if self.current_state in ["CAPTURING_PERFECTO", "CAPTURING_MALO"]:
                h, w, c = img.shape
                features = extraer_features(results.pose_landmarks.landmark)

                if features and len(features) == 99:
                    current_status_key = self.current_state.split('_')[1] 

                    self.data_features[current_status_key].append(features)
                    
                    time_elapsed = time.time() - self.capture_start_time
                    time_remaining = self.CAPTURE_DURATION - int(time_elapsed)
                    self.countdown_label.setText(f"Capturando... {time_remaining}s")

                    if time_elapsed >= self.CAPTURE_DURATION:
                        # Pasa a la siguiente etapa o finaliza
                        self.handle_stage_completion(current_status_key)
                        return 
                
        self.display_image(img)

    def handle_stage_completion(self, status):
        # Termina la etapa actual y pasa a la siguiente
        if status == "PERFECTO":
            self.set_state("READY_MALO")
        elif status == "MALO":
            self.set_state("FINISHED")
            self.save_data()

    def save_data(self):
        # Redirigir la salida de la consola para que los mensajes de guardado aparezcan en la GUI
        sys.stdout = self.console_redirect
        
        self.stage_label.setText("GUARDANDO DATOS...")
        self.stage_label.setStyleSheet("color: #00FF00; border: 1px solid #00FF00;")
        
        if self.data_features['PERFECTO'] and self.data_features['MALO']:
            
            data_to_save = {
                'PERFECTO': self.data_features['PERFECTO'],
                'MALO': self.data_features['MALO']
            }
            
            guardar_entrenamiento_bruto(self.nombre_perfil, data_to_save)
            
            QMessageBox.information(self, "Entrenamiento Exitoso", 
                                    f"Perfil '{self.nombre_perfil}' entrenado con éxito.")
            
            sys.stdout = sys.__stdout__ # Restaurar la salida original
            self.restart_app()

        else:
            sys.stdout = sys.__stdout__ # Restaurar la salida original
            QMessageBox.critical(self, "Error de Datos", "No se capturaron suficientes datos. Intente de nuevo.")
            self.restart_app()
            
    def restart_app(self):
        # Reiniciar la aplicación para un nuevo entrenamiento
        self.data_features = {'PERFECTO': [], 'MALO': []}
        self.nombre_perfil = None
        self.set_state("SELECT_PROFILE")
        self.stage_label.setText("Etapa Actual: Esperando Perfil...")
        self.camera_label.setText("Selecciona un perfil y haz clic en Iniciar.")


    def display_image(self, img):
        # Convertir OpenCV frame (BGR) a QImage (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()


# --- Función Principal ---
def run_trainer_gui():
    app = QApplication(sys.argv)
    window = PostureTrainerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_trainer_gui()