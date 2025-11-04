from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import mediapipe as mp
import numpy as np
import os

def detect_eye(request):
    result = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            image_path = f'media/{image_file.name}'
            with open(image_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Processamento com MediaPipe
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    iris_indices = [474, 475, 476, 477]
                    points = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                               int(face_landmarks.landmark[i].y * image.shape[0]))
                              for i in iris_indices]
                    if len(points) >= 2:
                        d = np.linalg.norm(np.array(points[0]) - np.array(points[2]))
                        result = f'Diâmetro da íris: {d:.2f} pixels'
    else:
        form = ImageUploadForm()
    return render(request, 'detector/upload.html', {'form': form, 'result': result})
