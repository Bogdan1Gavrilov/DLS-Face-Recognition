import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms

from models import StackedHourglassNetwork, FaceModel

# ПАРАМЕТРЫ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOURGLASS_IN = 256   # Размерность входа hourglass
EMBED_IN = 256       # Размерность входа модели классификации (получения эмбеддингов)
CONF_THR = 0.5       # Порог уверенности для модели детекции
NUM_KEYPOINTS = 5    # Число точек для предсказания hourglass
NUM_CLASSES = 1200   # Число классов в модели классификации
EMBEDDING_SIZE = 512 # Размерность эмбеддингов в модели классификации

# Пути к весам модели классификации, сохраняю сразу оба для того чтобы можно было наглядно показать и сравнить как модель идентифицирует лица
EMB_CE_PATH = r'weights\final_ce_model.pth'
EMB_ARC_PATH = r'weights\final_arc_model.pth'

transform = transforms.Compose([  # Нормализация фото перед подачей в Hourglass и перед подачей в модель с эмбеддингами одна и та же, как в Imagenet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Утилиты ----------
def extract_embedding(out, embedding_size=EMBEDDING_SIZE):
    #Извлекает эмбеддинги из вывода FaceModel.
    return out[1] if isinstance(out, tuple) else out

def get_coordinates_from_heatmaps(heatmaps_tensor, original_img_size=(256, 256)):
    device = heatmaps_tensor.device
    heatmaps = heatmaps_tensor.cpu().numpy()
    
    # Проверяем размерность
    if len(heatmaps.shape) == 4:  # [batch, 5, H, W]
        batch_size = heatmaps.shape[0]
        num_points = heatmaps.shape[1]
        h_heatmap, w_heatmap = heatmaps.shape[2], heatmaps.shape[3]
    elif len(heatmaps.shape) == 3:  # [5, H, W] - одиночный пример
        batch_size = 1
        num_points = heatmaps.shape[0]
        h_heatmap, w_heatmap = heatmaps.shape[1], heatmaps.shape[2]
        heatmaps = heatmaps.reshape(1, num_points, h_heatmap, w_heatmap)
    
    h_orig, w_orig = original_img_size[1], original_img_size[0]
    
    scale_x = w_orig / w_heatmap
    scale_y = h_orig / h_heatmap
    
    all_points = []
    for b in range(batch_size):
        points = []
        for i in range(num_points):  # 5 точек
            heatmap = heatmaps[b, i]
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            
            # Масштабируем к оригинальному размеру
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            
            points.append([x_scaled, y_scaled])
        all_points.append(points)
    
    points_array = np.array(all_points, dtype=np.float32)
    
    if len(heatmaps_tensor.shape) == 3:  # Если был одиночный пример
        return points_array[0]  # [5, 2]
    return points_array  # [batch_size, 5, 2]

def align_with_similarity(image, predicted_points, output_size=(256, 256)):
    #Преобразование подобия - тоже в своем роде афинное преобразование, но оно уже использует только две точки, а именно глаза.
    h, w = output_size[1], output_size[0]
    
    # Используем только глаза для определения поворота и масштаба
    left_eye = predicted_points[0]
    right_eye = predicted_points[1]
    
    # Целевые положения глаз
    target_left = np.array([0.34 * w, 0.46 * h])
    target_right = np.array([0.66 * w, 0.46 * h])
    
    # Вычисляем текущее и целевое расстояние между глазами
    current_eye_dist = np.linalg.norm(right_eye - left_eye)
    target_eye_dist = np.linalg.norm(target_right - target_left)
    
    # Масштаб
    scale = (target_eye_dist + 1e-7) / (current_eye_dist + 1e-7)
    
    #Находим разность между углами целевого и нынешнего векторов - именно на него будет повёрнуто изображение 
    current_vec = right_eye - left_eye #текущий вектор
    target_vec = target_right - target_left #целевой
    angle = np.degrees(np.arctan2(current_vec[1], current_vec[0]) - 
                       np.arctan2(target_vec[1], target_vec[0]))
    
    # Центр поворота - середина между глазами
    center = (left_eye + right_eye) / 2
    target_center = (target_left + target_right) / 2
    
    # Матрица преобразования
    M = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    
    # Корректируем смещение - необходимо чтобы поместить лица на изображениях в одно и то же место
    M[0, 2] += target_center[0] - center[0]
    M[1, 2] += target_center[1] - center[1]

    #И вновь финальным шагом обрабатываем изображения с теми же параметрами flags
    aligned = cv2.warpAffine(
        image, M, output_size,
        flags=cv2.INTER_LINEAR)
    
    return aligned, M

# Загрузка моделей
detection_model = YOLO(r'weights\yolov8n-face.pt')

points_model = StackedHourglassNetwork(num_stacks=2, num_keypoints=NUM_KEYPOINTS, upsample_outputs=False)
points_model.to(device)
ck = torch.load(r'weights\hourglass_model.pth', map_location=device, weights_only=True)
points_model.load_state_dict(ck['model_state_dict'])
points_model.eval()

# emb_model будет загружаться через функцию load_embedding_model (чтобы можно было переключать)
emb_model = None

def load_embedding_model(use_arcface: bool = False, ce_path: str = EMB_CE_PATH, arc_path: str = EMB_ARC_PATH):
    """
    Загружает emb_model глобально. Если use_arcface=True — создаётся FaceModel(use_arcface=True)
    и загружаются веса arc_path. Иначе — загружаются ce_path.
    """
    global emb_model
    emb_model = FaceModel(num_classes=NUM_CLASSES, embedding_size=EMBEDDING_SIZE, use_arcface=use_arcface)
    emb_model.to(device)
    path = arc_path if use_arcface else ce_path
    ck = torch.load(path, map_location=device, weights_only=True)
    emb_model.load_state_dict(ck['model_state_dict'])
    emb_model.eval()
    print(f"Загружена модель выдачи эмбеддингов: {'ArcFace' if use_arcface else 'CE'} из {path}")

# по умолчанию — CE
load_embedding_model(use_arcface=False)

@torch.no_grad()
def compute_embedding_from_image(img):
    if (img.shape[0], img.shape[1]) != (EMBED_IN, EMBED_IN):
        img = cv2.resize(img, (EMBED_IN, EMBED_IN), interpolation=cv2.INTER_LINEAR)
    inp = transform(img).unsqueeze(0).to(device)
    out = emb_model(inp)
    emb_tensor = extract_embedding(out, embedding_size=EMBEDDING_SIZE)
    emb = emb_tensor[0].cpu().numpy().astype(np.float32)
    return emb


@torch.no_grad()
def full_pipeline(path):
    img = cv2.imread(path)
    if img is None:
        return []
    H_img, W_img = img.shape[:2]
    results = detection_model(img, imgsz=640, conf=CONF_THR, iou=0.45, verbose=False)

    outs = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1 = max(0, min(x1, W_img-1)); y1 = max(0, min(y1, H_img-1))
            x2 = max(0, min(x2, W_img)); y2 = max(0, min(y2, H_img))
            w = x2 - x1; h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            conf = float(box.conf.item())

            # crop exactly by detection bbox (no margin)
            ex1, ey1, ex2, ey2 = x1, y1, x2, y2
            crop = img[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # Если детекция работала с BGR форматом, то мой Hourglass уже нуждается в классическом RGB
            crop_hg = cv2.resize(crop, (HOURGLASS_IN, HOURGLASS_IN), interpolation=cv2.INTER_LINEAR)

            hg_t = transform(crop_hg).unsqueeze(0).to(device)
            hg_outs = points_model(hg_t)
            last = hg_outs[-1]  # 1 x K x Hm x Wm
            coords_hg = get_coordinates_from_heatmaps(last, original_img_size=(HOURGLASS_IN, HOURGLASS_IN))
            if coords_hg.ndim == 3 and coords_hg.shape[0] == 1:
                coords_hg = coords_hg[0]

            # align and embedding
            aligned_img, _ = align_with_similarity(crop_hg, coords_hg, output_size=(EMBED_IN, EMBED_IN))
            emb = compute_embedding_from_image(aligned_img)
            outs.append({'bbox': (x1,y1,w,h,conf), 'embedding': emb})
    return outs

# Вспомогательная функция для кропа
def crop_img(path, bbox):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y, w, h = map(int, bbox[:4])
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(img.shape[1], x + w); y2 = min(img.shape[0], y + h)
    return img[y1:y2, x1:x2].copy()

# One-to-one greedy matching и отрисовка пар (показывает исходные фото перед парами)
def match_and_show_pairs(path1: str, path2: str, threshold: float = 0.5):
    res1 = full_pipeline(path1)
    res2 = full_pipeline(path2)

    if len(res1) == 0 or len(res2) == 0:
        print("На одном из изображений лиц не обнаружено")
        return []

    # матрица сходства
    E1 = np.stack([d['embedding'] for d in res1], axis=0)  # n1 x D
    E2 = np.stack([d['embedding'] for d in res2], axis=0)  # n2 x D
    sim = E1 @ E2.T

    # Собираем кандидатов выше threshold
    candidates = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            cos = float(sim[i, j])
            if cos >= threshold:
                candidates.append((cos, i, j))
    if not candidates:
        print("Совпадений не найдено")
        return []

    # Сортируем по убыванию cos
    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_i = set()
    matched_j = set()
    matches = []

    for cos, i, j in candidates:
        if i in matched_i or j in matched_j:
            continue
        matched_i.add(i)
        matched_j.add(j)
        matches.append((i, j, cos))

    # Печать
    for (i, j, cos) in matches:
        print(f"Совпадение: фотография 1 лицо №{i+1}  и  фотография 2 лицо № {j+1} | cosine = {cos:.4f}")

    # Сначала выводим исходные изображения
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # cv2 скачивает в BGR формате, поэтому приходится переводить в стандартный RGB
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(img1); plt.title("Фото 1"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(img2); plt.title("Фото 2"); plt.axis("off")
    plt.show()

    # Отрисовываем пары совпадений: для каждой пары — отдельный ряд (2 картинки бок о бок)
    for idx, (i, j, cos) in enumerate(matches):
        crop1 = crop_img(path1, res1[i]['bbox'])
        crop2 = crop_img(path2, res2[j]['bbox'])
        if crop1.size == 0 or crop2.size == 0:
            continue
        crop1 = cv2.resize(crop1, (EMBED_IN, EMBED_IN), interpolation=cv2.INTER_LINEAR)
        crop2 = cv2.resize(crop2, (EMBED_IN, EMBED_IN), interpolation=cv2.INTER_LINEAR)

        plt.figure(figsize=(6,3))
        plt.suptitle(f"Совпадение: фотография 1 лицо №{i+1}  и  фотография 2 лицо № {j+1} | cosine = {cos:.4f}",  fontsize=10)
        plt.subplot(1,2,1); plt.imshow(crop1); plt.title(f"img1 #{i}"); plt.axis('off')
        plt.subplot(1,2,2); plt.imshow(crop2); plt.title(f"img2 #{j}"); plt.axis('off')
        plt.show()

    return matches
