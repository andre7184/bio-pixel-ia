# detector/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
import cv2, os, mediapipe as mp
import numpy as np
from ultralytics import YOLO  # <-- NOVO: Importa o YOLO

# Nossos utils de análise
from .utils.iris_utils import diametro_iris_4p, diametro_iris_3p
from .utils.color_utils import detectar_cor_olhos, detectar_cor_cabelo
from .utils.image_utils import recortar_olho
from .utils.config import IRIS_MM, CALIBRACAO_ESCALA
from .utils.age_utils import faixa_etaria, estimar_idade

# --- CARREGA OS MODELOS (Fora da view, para eficiência) ---

# Carrega o modelo de estimativa de pose (YOLOv8 Nano-Pose)
# Ele será baixado automaticamente na primeira vez
try:
    yolo_model = YOLO('yolov8n-pose.pt')
except Exception as e:
    print(f"Erro ao carregar modelo YOLO: {e}. Certifique-se de ter 'ultralytics' instalado.")
    yolo_model = None

# Carrega o modelo de landmarks faciais (MediaPipe)
try:
    mp_face_mesh = mp.solutions.face_mesh
    fm = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
except Exception as e:
    print(f"Erro ao carregar MediaPipe FaceMesh: {e}")
    fm = None

# -----------------------------------------------------------

def detect_height(request):
    pessoas = []
    error_message = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid() and yolo_model and fm:
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

            # --- 1. ETAPA YOLO: DETECTAR TODOS OS CORPOS ---
            # Aqui está a mágica: o YOLO processa a imagem UMA vez
            # e retorna uma lista de TODAS as pessoas detectadas.
            pose_results = yolo_model(image)

            # pose_results[0] contém os dados da primeira imagem (nós só enviamos uma)
            if not pose_results[0].keypoints:
                error_message = "Nenhuma pessoa foi detectada na imagem pelo YOLO."
            else:
                # Pega os keypoints (landmarks da pose) e caixas (bounding boxes)
                keypoints_list = pose_results[0].keypoints.cpu().numpy()
                boxes_list = pose_results[0].boxes.cpu().numpy()

                # --- 2. LOOP "CORPOS PRIMEIRO" ---
                # Agora, fazemos um loop por cada pessoa que o YOLO encontrou
                for idx, (person_kpts, person_box) in enumerate(zip(keypoints_list.data, boxes_list.xyxy)):
                    
                    # --- 2A. CALCULAR ALTURA EM PIXELS (da Pose) ---
                    # Índices dos keypoints do YOLOv8-Pose (COCO)
                    # 0=nariz, 1=olho_esq, 2=olho_dir, 3=orelha_esq, 4=orelha_dir
                    # 15=calcanhar_esq, 16=calcanhar_dir
                    
                    try:
                        # Pega pontos Y da cabeça (se confiança > 0.1)
                        head_points_y = [person_kpts[j][1] for j in [0, 1, 2, 3, 4] if person_kpts[j][2] > 0.1] 
                        # Pega pontos Y do calcanhar (se confiança > 0.1)
                        heel_points_y = [person_kpts[j][1] for j in [15, 16] if person_kpts[j][2] > 0.1]
                        
                        if not head_points_y or not heel_points_y:
                            continue # Ignora pessoa se não tiver cabeça ou calcanhar visível

                        y_min = min(head_points_y) # Ponto mais alto da cabeça
                        y_max = max(heel_points_y) # Ponto mais baixo do calcanhar
                        
                        altura_pixels = abs(y_max - y_min)
                        
                        # Desenha a linha da altura na imagem de cópia
                        cx = int(person_kpts[0][0]) # Centraliza a linha no nariz da pessoa
                        cv2.line(image_copy, (cx, int(y_min)), (cx, int(y_max)), (0, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"Erro ao calcular altura pixels para pessoa {idx}: {e}")
                        continue # Pula para a próxima pessoa
                        
                    # --- 2B. OBTER RECORTE DO ROSTO (da Pose) ---
                    # Cria um 'face_crop' usando os keypoints da cabeça
                    face_x = [person_kpts[j][0] for j in [0, 1, 2, 3, 4] if person_kpts[j][2] > 0.1]
                    face_y = [person_kpts[j][1] for j in [0, 1, 2, 3, 4] if person_kpts[j][2] > 0.1]

                    if not face_x or not face_y:
                        continue # Pula se não houver pontos de rosto válidos

                    x1, x2 = max(0, int(min(face_x)) - 20), min(iw, int(max(face_x)) + 20)
                    y1, y2 = max(0, int(min(face_y)) - 20), min(ih, int(max(face_y)) + 40) # Dá mais espaço para baixo
                    
                    face_crop = image[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        continue # Pula se o recorte do rosto falhar
                        
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    
                    # --- 2C. CALCULAR ESCALA (mm/px) (do Rosto) ---
                    # Roda o FaceMesh APENAS no recorte do rosto
                    fr = fm.process(face_rgb)
                    
                    escala = None
                    diam_d = diam_e = diff = None
                    cor_olhos = "N/A"
                    eye_right_url = eye_left_url = None
                    
                    if fr.multi_face_landmarks:
                        lm_face = fr.multi_face_landmarks[0].landmark
                        crop_h, crop_w = face_crop.shape[:2]
                        
                        # Pontos da íris relativos ao face_crop
                        iris_d_rel = [(int(lm.x * crop_w), int(lm.y * crop_h)) for lm in [lm_face[i] for i in [474,475,476,477]]]
                        iris_e_rel = [(int(lm.x * crop_w), int(lm.y * crop_h)) for lm in [lm_face[i] for i in [469,470,471,472]]]
                        
                        # Calcula diâmetro da íris
                        d4_d, ok_d = diametro_iris_4p(iris_d_rel)
                        diam_d = d4_d if ok_d else diametro_iris_3p([iris_d_rel[0], iris_d_rel[2], iris_d_rel[3]])
                        
                        d4_e, ok_e = diametro_iris_4p(iris_e_rel)
                        diam_e = d4_e if ok_e else diametro_iris_3p([iris_e_rel[0], iris_e_rel[2], iris_e_rel[3]])

                        if diam_d and diam_e:
                            d_medio = (diam_d + diam_e) / 2
                            escala = (IRIS_MM / d_medio) * CALIBRACAO_ESCALA
                            diff = abs(diam_d - diam_e) / d_medio * 100
                    
                        # Detectar cor dos olhos (usa o crop e os pontos relativos)
                        cor_olhos = detectar_cor_olhos(face_crop, iris_d_rel)
                        
                        # Recortes dos olhos
                        eye_right_url = recortar_olho(face_crop, iris_d_rel, f"eye_right_{idx}", media_dir, image_name)
                        eye_left_url = recortar_olho(face_crop, iris_e_rel, f"eye_left_{idx}", media_dir, image_name)

                    # --- 2D. CALCULAR IDADE E COR (do Rosto) ---
                    idade_estimativa = estimar_idade(face_crop)
                    faixa = faixa_etaria(idade_estimativa)
                    cor_cabelo = detectar_cor_cabelo(face_crop)
                    
                    # --- 2E. COMBINAR TUDO ---
                    altura_cm = None
                    if escala and altura_pixels:
                        altura_mm = altura_pixels * escala
                        altura_cm = altura_mm / 10.0
                        
                    # --- 2F. SALVAR IMAGENS DE RECORTE ---
                    face_name = f'face_{idx}_{image_name}'
                    cv2.imwrite(os.path.join(media_dir, face_name), face_crop)
                    face_url = f'/media/{face_name}'
                    
                    pessoa = {
                        "id": idx + 1,
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
                        "eye_right_url": eye_right_url,
                        "eye_left_url": eye_left_url
                    }
                    pessoas.append(pessoa)
                
                # Salva a imagem final com todas as linhas de altura
                body_name = f'body_all_{image_name}'
                cv2.imwrite(os.path.join(media_dir, body_name), image_copy)
                
                # Adiciona a imagem com as marcações à primeira pessoa (para exibição no template)
                if pessoas:
                    pessoas[0]['body_url'] = f'/media/{body_name}'

        elif not (yolo_model and fm):
             error_message = "Erro: Modelos de IA não foram carregados corretamente."
    else:
        form = ImageUploadForm()

    return render(request, 'detector/upload.html', {
        'form': form,
        'pessoas': pessoas,
        'error_message': error_message
    })