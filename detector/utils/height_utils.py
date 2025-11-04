# detector/utils/height_utils.py
import cv2
import mediapipe as mp
import numpy as np

# Inicialize a solução Pose fora da função para ser mais eficiente
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

def medir_altura_pixels(image_bgr: np.ndarray):
    """
    Detecta UMA pessoa principal na imagem e retorna sua altura em pixels
    e as coordenadas Y da cabeça e calcanhar.
    """
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None  # Nenhuma pessoa detectada

    lm = results.pose_landmarks.landmark

    # Pontos do corpo para altura
    # Ponto 10 é um bom ponto médio para o topo da cabeça
    # Pontos 29 (calcanhar esquerdo) e 30 (calcanhar direito)
    y_min = int(lm[10].y * h)
    y_max = max(int(lm[29].y * h), int(lm[30].y * h))
    
    # Validação simples (se a pessoa estiver de cabeça para baixo, por exemplo)
    if y_max <= y_min:
        return None

    altura_px = y_max - y_min
    
    return {
        "altura_pixels": altura_px,
        "y_min_head": y_min,
        "y_max_heel": y_max,
        "landmarks": lm # Retorna todos os landmarks se precisar de mais dados
    }