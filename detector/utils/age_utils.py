# detector/utils/age_utils.py

import os
import cv2
import numpy as np
import onnxruntime as ort

# Resolve caminho absoluto para o diretório raiz do projeto (onde está manage.py)
# __file__ -> detector/utils/age_utils.py
# BASE_DIR -> bio-pixel-ia/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_googlenet.onnx")

# Inicializa a sessão ONNX uma única vez
# Se precisar rodar em GPU com onnxruntime-gpu, ajuste providers.
_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Buckets de idade do modelo age_googlenet.onnx
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# Mapeia o bucket para uma idade média aproximada (inteiro)
AGE_BUCKET_MEAN = {
    "(0-2)": 1,
    "(4-6)": 5,
    "(8-12)": 10,
    "(15-20)": 18,
    "(25-32)": 28,
    "(38-43)": 40,
    "(48-53)": 50,
    "(60-100)": 70,
}


def _preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Converte BGR -> RGB, redimensiona para 224x224 e normaliza para o modelo ONNX.
    Retorna tensor com shape (1, 3, 224, 224), float32, normalizado em [0,1].
    """
    img = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = img.transpose(2, 0, 1)[None, ...] / 255.0  # (1, 3, 224, 224)
    return blob


def estimar_idade(face_crop: np.ndarray) -> int:
    """
    Estima idade aproximada a partir de um recorte de rosto (BGR).
    Retorna um inteiro (idade média do bucket predito).
    """
    try:
        # Valida entrada
        if face_crop is None or face_crop.size == 0:
            return 12  # fallback

        blob = _preprocess_face(face_crop)
        inputs = {_session.get_inputs()[0].name: blob}
        outputs = _session.run(None, inputs)
        probs = outputs[0][0]  # vetor de 8 classes

        idx = int(np.argmax(probs))
        faixa = AGE_BUCKETS[idx]
        idade = AGE_BUCKET_MEAN.get(faixa, 30)

        # Proteções simples contra outliers
        idade = int(max(0, min(100, idade)))
        return idade
    except Exception:
        # Em caso de erro na inferência, devolve um valor conservador
        return 12


def faixa_etaria(idade_estimativa: int) -> str:
    """
    Classifica a idade estimada em uma faixa etária amigável.
    """
    if idade_estimativa <= 2:
        return "Bebê (0–2)"
    elif idade_estimativa <= 12:
        return "Criança (2–12)"
    elif idade_estimativa <= 18:
        return "Adolescente (12–18)"
    elif idade_estimativa <= 30:
        return "Jovem Adulto (18–30)"
    elif idade_estimativa <= 40:
        return "Adulto (30–40)"
    elif idade_estimativa <= 50:
        return "Adulto (40–50)"
    elif idade_estimativa <= 60:
        return "Adulto (50–60)"
    else:
        return "Idoso (>60)"
