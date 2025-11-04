import cv2
import os

def recortar_olho(face_crop, pontos, nome, media_dir, image_name):
    x_min = max(0, min([p[0] for p in pontos]) - 5)
    x_max = min(face_crop.shape[1], max([p[0] for p in pontos]) + 5)
    y_min = max(0, min([p[1] for p in pontos]) - 5)
    y_max = min(face_crop.shape[0], max([p[1] for p in pontos]) + 5)
    eye_crop = face_crop[y_min:y_max, x_min:x_max].copy()
    for p in pontos:
        cv2.circle(eye_crop, (p[0]-x_min, p[1]-y_min), 2, (0,0,255), -1)
    eye_name = f'{nome}_{image_name}'
    cv2.imwrite(os.path.join(media_dir, eye_name), eye_crop)
    return f'/media/{eye_name}'
