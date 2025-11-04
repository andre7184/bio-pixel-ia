import cv2
import numpy as np

def detectar_cor_olhos(face_crop, iris_points):
    if not iris_points or len(iris_points) < 2:
        return "Não detectado"
    mask = np.zeros(face_crop.shape[:2], dtype=np.uint8)
    centro = np.mean(iris_points, axis=0).astype(int)
    raio = int(np.linalg.norm(np.array(iris_points[0]) - np.array(iris_points[1]))/2)
    raio = int(raio * 0.6)
    cv2.circle(mask, tuple(centro), raio, 255, -1)
    if np.count_nonzero(mask) < 25:
        return "Não detectado"
    iris_pixels = cv2.bitwise_and(face_crop, face_crop, mask=mask)
    hsv = cv2.cvtColor(iris_pixels, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vals_h = h[mask == 255]; vals_s = s[mask == 255]; vals_v = v[mask == 255]
    valid = (vals_v > 30) & (vals_v < 230)
    if np.count_nonzero(valid) == 0:
        return "Não detectado"
    h_m = float(np.median(vals_h[valid])); s_m = float(np.median(vals_s[valid]))
    if 180 <= h_m <= 240 and s_m > 40: return "Azuis"
    elif 60 <= h_m <= 120 and s_m > 40: return "Verdes"
    elif 20 <= h_m <= 50 and s_m > 60: return "Mel/Âmbar"
    else: return "Castanhos/Preto"

def detectar_cor_cabelo(face_crop):
    h, w = face_crop.shape[:2]
    hair_region = face_crop[0:int(h*0.2), :]
    hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
    h_m = np.mean(hsv[:,:,0]); s_m = np.mean(hsv[:,:,1]); v_m = np.mean(hsv[:,:,2])
    if v_m < 60: return "Preto"
    elif 10 <= h_m <= 30 and s_m > 90: return "Ruivo"
    elif v_m > 180 and s_m < 80: return "Loiro"
    else: return "Castanho"
