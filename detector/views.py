# detector/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
import cv2, os, mediapipe as mp
import numpy as np
from .utils.iris_utils import diametro_iris_4p, diametro_iris_3p
from .utils.color_utils import detectar_cor_olhos, detectar_cor_cabelo
from .utils.image_utils import recortar_olho
from .utils.config import IRIS_MM, CALIBRACAO_ESCALA
from .utils.age_utils import faixa_etaria, estimar_idade
from .utils.height_utils import medir_altura_pixels # <-- NOSSA NOVA FUNÇÃO

# Inicializa as soluções do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def detect_height(request):
    pessoas = []
    error_message = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            image_name = image_file.name
            media_dir = 'media'
            os.makedirs(media_dir, exist_ok=True)
            input_path = os.path.join(media_dir, image_name)
            
            with open(input_path, 'wb+') as f:
                for chunk in image_file.chunks(): f.write(chunk)

            image = cv2.imread(input_path)
            image_copy = image.copy() # Cópia para desenhar
            ih, iw = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --- 1. DETECTAR ALTURA (POSE) PRIMEIRO ---
            pose_data = medir_altura_pixels(image)

            if not pose_data:
                error_message = "Nenhum corpo de pessoa foi detectado na imagem."
            else:
                altura_pixels = pose_data['altura_pixels']
                escala = None
                altura_cm = None
                
                # --- 2. DETECTAR ÍRIS (FACE MESH) PARA ESCALA ---
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
                    fr = fm.process(rgb)

                if fr.multi_face_landmarks:
                    lm_face = fr.multi_face_landmarks[0].landmark
                    
                    # Usa o IW, IH da imagem original para calcular os pontos da íris
                    iris_d = [(int(lm_face[i].x*iw), int(lm_face[i].y*ih)) for i in [474,475,476,477]]
                    iris_e = [(int(lm_face[i].x*iw), int(lm_face[i].y*ih)) for i in [469,470,471,472]]

                    # Olho direito
                    d4_d, ok_d = diametro_iris_4p(iris_d)
                    diam_d = d4_d if ok_d else diametro_iris_3p([iris_d[0], iris_d[2], iris_d[3]])

                    # Olho esquerdo
                    d4_e, ok_e = diametro_iris_4p(iris_e)
                    diam_e = d4_e if ok_e else diametro_iris_3p([iris_e[0], iris_e[2], iris_e[3]])

                    if diam_d and diam_e:
                        d_medio = (diam_d + diam_e) / 2
                        escala = (IRIS_MM / d_medio) * CALIBRACAO_ESCALA
                        diff = abs(diam_d - diam_e) / d_medio * 100

                        # --- 3. CALCULAR ALTURA REAL ---
                        altura_mm = altura_pixels * escala
                        altura_cm = altura_mm / 10.0
                
                if not escala:
                    error_message = "Corpo detectado, mas não foi possível localizar a íris para calcular a escala."

                # --- 4. DETECTAR ROSTO (FACE DETECTION) PARA IDADE/COR ---
                # Usamos um detector de rosto separado para garantir um bom "crop" para os modelos de idade/cor
                face_crop = None
                face_url = None
                idade_estimativa = 12 # Padrão
                faixa = "N/A"
                cor_olhos = "N/A"
                cor_cabelo = "N/A"
                
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
                    detections = fd.process(rgb)
                    if detections.detections:
                        det = detections.detections[0] # Pega o primeiro rosto
                        bbox = det.location_data.relative_bounding_box
                        x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
                        face_crop = image[y:y+h, x:x+w].copy()

                        # Salva o rosto
                        face_name = f'face_0_{image_name}'
                        cv2.imwrite(os.path.join(media_dir, face_name), face_crop)
                        face_url = f'/media/{face_name}'
                        
                        # Agora podemos rodar os utils com segurança
                        idade_estimativa = estimar_idade(face_crop)
                        faixa = faixa_etaria(idade_estimativa)
                        cor_cabelo = detectar_cor_cabelo(face_crop)
                        # A detecção de cor de olhos idealmente usa o crop do FaceMesh (lm_face), não o crop do FaceDetection
                        # Mas para simplificar, vamos usar a íris_d que já temos
                        cor_olhos = detectar_cor_olhos(image, iris_d) # Passa a imagem inteira e os pontos


                # --- 5. DESENHAR E SALVAR RESULTADOS ---
                # Desenha a linha da altura na cópia da imagem
                cx = int(iw * 0.5) # Linha no meio
                cv2.line(image_copy, (cx, pose_data['y_min_head']), (cx, pose_data['y_max_heel']), (0,255,255), 3)
                
                body_name = f'body_0_{image_name}'
                cv2.imwrite(os.path.join(media_dir, body_name), image_copy)
                body_url = f'/media/{body_name}'

                pessoa = {
                    "id": 1,
                    "iris_direita": f"{diam_d:.2f}px" if diam_d else "Falha",
                    "iris_esquerda": f"{diam_e:.2f}px" if diam_e else "Falha",
                    "media_iris": f"{(diam_d+diam_e)/2:.2f}px" if diam_d and diam_e else "N/A",
                    "escala": f"{escala:.3f} mm/px" if escala else "N/A",
                    "diff_iris": f"{diff:.1f}%" if diff else "N/A",
                    "cor_olhos": cor_olhos,
                    "cor_cabelo": cor_cabelo,
                    "altura": f"{altura_cm:.1f} cm" if altura_cm else "N/A",
                    "idade_estimativa": f"{idade_estimativa} anos",
                    "faixa_etaria": faixa,
                    "face_url": face_url,
                    "body_url": body_url,
                }
                pessoas.append(pessoa)

    else:
        form = ImageUploadForm()

    return render(request, 'detector/upload.html', {
        'form': form,
        'pessoas': pessoas,
        'error_message': error_message
    })