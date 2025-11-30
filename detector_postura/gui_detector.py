import cv2
import sys
import warnings
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import deque

# Importaciones de PyQt (QTextCursor corregido a QtGui)
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QTextCursor # <-- CORRECCIÓN FINAL

# Importaciones de la Lógica del Motor
from posture_logic import mp_pose, mp_drawing, pose, obtener_nombres_de_perfiles, extraer_features, clasificar_postura, cargar_datos_brutos_para_recalculo, PredictionFilter, disparar_alarma_interruptible, detener_alarma

warnings.filterwarnings("ignore")

# --- FUNCIONES DE UTILIDAD DE TIEMPO ---

def format_time(seconds):
    """Convierte segundos en formato H:MM:SS."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:01d}:{m:02d}:{s:02d}"

# --- CLASE DE REDIRECCIÓN DE CONSOLA ---
class ConsoleRedirect(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass

# --- CLASE DE ENTRENAMIENTO ASÍNCRONO ---
class TrainerThread(QThread):
    """Hilo para entrenar el modelo ML sin congelar la GUI."""
    training_finished = pyqtSignal(object)
    training_error = pyqtSignal(str)
    
    def __init__(self, nombre_perfil, gui_logger):
        super().__init__()
        self.nombre_perfil = nombre_perfil
        self._gui_logger = gui_logger
        

    def run(self):
        # Redirigir temporalmente la salida para capturar logs de ML
        sys.stdout = self._gui_logger
        
        try:
            modelo_rf = self._entrenar_modelo_rf(self.nombre_perfil)
            
            # Restaurar la salida original
            sys.stdout = sys.__stdout__ 
            
            if modelo_rf:
                self.training_finished.emit(modelo_rf)
            else:
                self.training_error.emit("No hay suficientes datos para entrenar.")
        except Exception as e:
            # Restaurar la salida original incluso en caso de error
            sys.stdout = sys.__stdout__ 
            self.training_error.emit(f"Error durante el entrenamiento: {e}")

    def _entrenar_modelo_rf(self, nombre_perfil):
        """Carga todos los datos brutos, prepara el dataset y entrena el Random Forest."""
        print("\n--- INICIANDO ENTRENAMIENTO ML ---")
        datos_brutos = cargar_datos_brutos_para_recalculo(nombre_perfil)
        
        if not datos_brutos['PERFECTO'] or not datos_brutos['MALO']:
            return None
            
        X_perfecto = np.array(datos_brutos['PERFECTO'])
        X_malo = np.array(datos_brutos['MALO'])
        
        if len(X_perfecto) == 0 or len(X_malo) == 0:
            return None

        y_perfecto = np.array(['PERFECTO'] * len(X_perfecto))
        y_malo = np.array(['MALO'] * len(X_malo))
        
        X = np.concatenate([X_perfecto, X_malo])
        y = np.concatenate([y_perfecto, y_malo])
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"[ML] Entrenamiento completado. Precisión en el dataset de entrenamiento: {accuracy:.2f}")
        
        return model

# --- CLASE DE LA VENTANA PRINCIPAL ---

class PostureDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Redirigir la salida de la consola
        self.console_redirect = ConsoleRedirect()
        self.console_redirect.text_written.connect(self.update_log)
        
        self.setWindowTitle("DETECTOR DE POSTURA DE USO DIARIO")
        self.setGeometry(100, 100, 1000, 650)
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        # Estado del detector
        self.modelo_rf = None
        self.cap = None
        self.prediction_filter = PredictionFilter(window_size=15)
        self.trainer_thread = None
        self.selected_profile = None

        # --- VARIABLES DE CONTEO DE TIEMPO ---
        self.tiempo_bueno_total = 0.0
        self.tiempo_malo_total = 0.0
        self.last_frame_time = time.time()
        
        self.setup_ui()
        self.load_profiles_menu()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Panel Izquierdo (Control, Métricas y Log)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. Menú de Perfiles y Botón de Inicio
        self.profile_combo = QComboBox()
        self.profile_combo.setStyleSheet("color: black; background-color: white; padding: 5px;")
        self.profile_combo.setFont(QFont("Arial", 10))
        
        left_panel.addWidget(QLabel("SELECCIONA UN PERFIL:"))
        left_panel.addWidget(self.profile_combo)

        self.start_button = QPushButton("EJECUTAR APLICACION")
        self.start_button.setStyleSheet("background-color: #007ACC; color: white; padding: 10px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_detection)
        left_panel.addWidget(self.start_button)
        left_panel.addSpacing(15)

        # 2. Indicadores de Tiempo (Métricas)
        self.time_good_label = QLabel("Tiempo con una BUENA POSTURA:\n0:00:00")
        self.time_good_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.time_good_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_good_label.setStyleSheet("color: #00FF00; padding: 10px; border: 2px solid #00AA00; border-radius: 5px;")
        left_panel.addWidget(self.time_good_label)

        self.time_bad_label = QLabel("Tiempo con una MALA POSTURA:\n0:00:00")
        self.time_bad_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.time_bad_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_bad_label.setStyleSheet("color: #FF0000; padding: 10px; border: 2px solid #AA0000; border-radius: 5px;")
        left_panel.addWidget(self.time_bad_label)

        # 3. Indicador de Predicción (Retroalimentación Principal)
        self.feedback_label = QLabel("Iniciando Detector...")
        self.feedback_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback_label.setStyleSheet("color: #FFFFFF; padding: 20px; border-radius: 5px; background-color: #333333;")
        left_panel.addWidget(self.feedback_label)
        
        # 4. Área de Log / Consola
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setStyleSheet("background-color: #000000; color: #FFFFFF; border: 1px solid #555;")
        left_panel.addWidget(QLabel("Log del Sistema:"))
        left_panel.addWidget(self.log_widget)
        
        control_widget = QWidget()
        control_widget.setLayout(left_panel)
        control_widget.setFixedWidth(420)
        main_layout.addWidget(control_widget)

        # Panel Derecho (Cámara)
        self.camera_label = QLabel("Cargando Cámara...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; border: 2px solid #555555;")
        self.camera_label.setMinimumSize(700, 500)
        main_layout.addWidget(self.camera_label)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def update_log(self, text):
        """Añade texto al widget de log."""
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_widget.setTextCursor(cursor)

    def load_profiles_menu(self):
        self.profile_combo.clear()
        self.profile_combo.addItem("Perfiles")
        self.profile_combo.addItems(obtener_nombres_de_perfiles())

    def start_detection(self):
        profile_name = self.profile_combo.currentText()
        if profile_name == "Perfiles":
            self.feedback_label.setText("Selecciona un perfil")
            self.feedback_label.setStyleSheet("color: yellow; padding: 20px; background-color: #444400;")
            return
        
        self.selected_profile = profile_name
        self.start_button.setText("Entrenando Modelo...")
        self.start_button.setEnabled(False)
        self.feedback_label.setText("Cargando datos...")

        # Reiniciar contadores de tiempo
        self.tiempo_bueno_total = 0.0
        self.tiempo_malo_total = 0.0
        self.last_frame_time = time.time()

        # Iniciar entrenamiento en hilo separado
        self.trainer_thread = TrainerThread(self.selected_profile, self.console_redirect)
        self.trainer_thread.training_finished.connect(self.on_training_finished)
        self.trainer_thread.training_error.connect(self.on_training_error)
        self.trainer_thread.start()

    def on_training_finished(self, model):
        self.modelo_rf = model
        
        # Iniciar cámara y timer
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_label.setText("ERROR: Cámara no disponible.")
            self.start_button.setEnabled(True)
            return

        self.timer.start(30) # ~33 FPS
        self.start_button.setText("DETECCIÓN ACTIVA")
        self.start_button.setEnabled(False)
        self.feedback_label.setText("POSTURA OK")

    def on_training_error(self, message):
        self.camera_label.setText("ERROR ML: " + message)
        self.start_button.setText("REINTENTAR")
        self.start_button.setEnabled(True)
        self.feedback_label.setText("ERROR ML")
        detener_alarma()

    def update_frame(self):
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        success, img = self.cap.read()
        if not success:
            return

        img = cv2.flip(img, 1) 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        posture_text, color_rgb = "Buscando...", (255, 255, 255)

        if results.pose_landmarks and self.modelo_rf:
            
            # 1. Extracción de Features
            features = extraer_features(results.pose_landmarks.landmark)
            
            if features and len(features) == 99 and features[0] != 0.0:
                # Predicción y Filtro
                X_input = np.array(features).reshape(1, -1)
                prediction = self.modelo_rf.predict(X_input)[0]
                
                self.prediction_filter.add_prediction(prediction)
                smoothed_prediction = self.prediction_filter.get_dominant_prediction()
                
                posture_text, color_rgb = clasificar_postura(smoothed_prediction)
                
                # 2. Lógica de Conteo de Tiempo y Alarma
                is_currently_bad = (smoothed_prediction == 'MALO')
                
                if is_currently_bad:
                    # Alarma ON y Conteo MALO
                    disparar_alarma_interruptible()
                    self.tiempo_malo_total += delta_time 
                else:
                    # Alarma OFF y Conteo BUENO/ACEPTABLE
                    detener_alarma()
                    if smoothed_prediction == 'PERFECTO':
                        self.tiempo_bueno_total += delta_time
                    elif smoothed_prediction == 'ACEPTABLE':
                        self.tiempo_bueno_total += delta_time 
                
                # Dibujar landmarks de OpenCV
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # 3. Actualizar la GUI
        self.update_metrics_and_feedback(posture_text, color_rgb)
        self.display_image(img)

    def update_metrics_and_feedback(self, text, rgb_color):
        # RGB a Color de PyQt
        q_color = QColor(rgb_color[0], rgb_color[1], rgb_color[2])
        
        # Actualizar Tiempos
        self.time_good_label.setText(f"Tiempo con una BUENA POSTURA:\n{format_time(self.tiempo_bueno_total)}")
        self.time_bad_label.setText(f"Tiempo con una MALA POSTURA:\n{format_time(self.tiempo_malo_total)}")

        # Actualizar Feedback
        self.feedback_label.setText(text)
        self.feedback_label.setStyleSheet(f"color: white; padding: 20px; border-radius: 5px; background-color: {q_color.darker(150).name()}; border: 2px solid {q_color.name()};")
        
    def display_image(self, img):
        # Convertir OpenCV frame (BGR) a QImage (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Escalar imagen para ajustarse a la etiqueta
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        # Detener la cámara y el timer al cerrar
        detener_alarma()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    # Redirigir la salida de la consola al inicio de la aplicación
    app = QApplication(sys.argv)
    window = PostureDetectorApp()
    window.show()
    sys.exit(app.exec())