**Este programa es un Detector de Postura Óptima (ML Powered).**

# Utiliza Inteligencia Artificial (Machine Learning) para analizar en tiempo real las coordenadas 3D de tu cuerpo a través de la cámara. Su propósito es monitorizar tu ergonomía mientras trabajas, clasificando tu postura como 'Postura Óptima' o 'Riesgo Ergonómico'. El sistema acumula el tiempo que pasas en cada estado y te alerta con una alarma sonora instantánea si detecta una mala postura sostenida.

🛠️ Comandos Necesarios

Debes ejecutar los comandos en tu terminal MINGW64 o Git Bash con el entorno virtual (venv) activo.

**Fase 1: Configuración**

a) Crear el Entorno Virtual (venv):
python -m venv venv

b) Activar el Entorno Virtual (venv):
source venv/Scripts/activate

c) Instalar Librerías (Solo si no lo has hecho):
pip install opencv-python mediapipe numpy scikit-learn PyQt6

**Fase 2: Entrenamiento (Captura de Datos)**
El sistema necesita aprender tu postura única. Debes hacer esto al menos una vez por perfil.

python gui_trainer.py

- Instrucciones al correr gui_trainer.py:

Aparecerá una ventana de PyQt.
Haz clic en SELECCIONAR / CREAR PERFIL (ej., nombra el perfil Escritorio).
Sigue las instrucciones en pantalla para capturar 10 segundos de tu Postura IDEAL y 10 segundos de tu Postura PELIGROSA (encorvado).
Esto guardará los datos necesarios en la carpeta /PERFILES.

**Fase 3: Ejecución Diaria (Detección)**
Una vez que el perfil esté entrenado, usa este comando para el uso diario:
python gui_detector.py

Instrucciones al correr gui_detector.py:

Aparecerá la interfaz principal.
Selecciona el perfil que entrenaste (Escritorio).
Haz clic en INICIAR ANÁLISIS.
El modelo ML se entrena instantáneamente con tus datos y comienza a monitorear el Tiempo Productivo y el Tiempo de Riesgo.
