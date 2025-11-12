import os
import cv2
import numpy as np

# Répertoire des images collectées
DATA_DIR = './data'
# Répertoire pour les images augmentées
AUGMENTED_DATA_DIR = './augmented_data'
if not os.path.exists(AUGMENTED_DATA_DIR):
    os.makedirs(AUGMENTED_DATA_DIR)

def augment_image(image):
    rows, cols = image.shape[:2]
    augmented_images = []
    
    # Rotation
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
    
    # Translation
    for dx, dy in [(10, 10), (-10, -10)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(translated)
    
    # Flipping
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Scaling
    for scale in [1.2, 0.8]:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if scaled.shape[0] > rows:
            scaled = cv2.resize(scaled, (cols, rows))
        imgWhite = np.ones_like(image) * 255
        x_center = (imgWhite.shape[1] - scaled.shape[1]) // 2
        y_center = (imgWhite.shape[0] - scaled.shape[0]) // 2
        imgWhite[y_center:y_center+scaled.shape[0], x_center:x_center+scaled.shape[1]] = scaled
        augmented_images.append(imgWhite)
    
    return augmented_images

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    augmented_class_path = os.path.join(AUGMENTED_DATA_DIR, class_dir)
    if not os.path.exists(augmented_class_path):
        os.makedirs(augmented_class_path)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)
        
        augmented_images = augment_image(image)
        
        for idx, aug_img in enumerate(augmented_images):
            aug_img_name = f'{os.path.splitext(img_name)[0]}_aug_{idx}.jpg'
            cv2.imwrite(os.path.join(augmented_class_path, aug_img_name), aug_img)

        cv2.imwrite(os.path.join(augmented_class_path, img_name), image)
