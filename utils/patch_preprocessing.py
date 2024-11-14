import os
import numpy as np
import cv2
import openslide
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import streamlit as st

# Transformación básica
basic_augmentations = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

# Función para verificar si un parche es válido
def is_valid_patch(img, threshold=15):
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.std(grayscale) >= threshold

# Función para verificar si un parche es fondo
def is_background_patch(img, std_threshold=15, mean_diff_threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.std(gray) < std_threshold:
        return True
    mean_r, mean_g, mean_b = img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean()
    return max(abs(mean_r - mean_g), abs(mean_r - mean_b), abs(mean_g - mean_b)) < mean_diff_threshold

# Función para convertir la imagen TIFF a un formato estándar
def convert_to_standard_tiff(image_path):
    try:
        img = Image.open(image_path)
        standard_path = image_path.replace(".tif", "_standard.tif")
        img.save(standard_path, format="TIFF", compression=None)
        print(f"Imagen convertida a formato TIFF estándar: {standard_path}")
        return standard_path
    except Exception as e:
        print(f"Error al convertir la imagen: {e}")
        return None

# Función principal de preprocesamiento para una sola imagen usando openslide
def patch_preprocessing(image_path, num_patches=15):
    # Intenta abrir la imagen con openslide
    try:
        try:
            slide = openslide.OpenSlide(image_path)
        except (openslide.OpenSlideUnsupportedFormatError, openslide.OpenSlideError):
            print(f"Error al abrir el archivo TIFF original: {image_path}")
            print("Intentando convertir el archivo a un formato estándar...")
            # Si falla, convierte la imagen a un formato TIFF estándar
            image_path = convert_to_standard_tiff(image_path)
            if not image_path:
                print("No se pudo convertir la imagen. Abortando preprocesamiento.")
                return None
            slide = openslide.OpenSlide(image_path)

        width, height = slide.dimensions
        patches = []

        patch_size = 512
        step_size = patch_size // 2
        max_attempts = num_patches * 30
        attempts = 0

        for y in range(0, height - patch_size + 1, step_size):
            for x in range(0, width - patch_size + 1, step_size):
                if len(patches) >= num_patches:
                    break

                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                patch = np.array(patch)

                if is_background_patch(patch):
                    continue

                if is_valid_patch(patch):
                    augmented = basic_augmentations(image=patch)
                    patches.append(augmented['image'])
                    attempts = 0
                else:
                    attempts += 1

                if attempts >= max_attempts:
                    print(f"Maximum attempts exceeded for image {image_path}")
                    break

        if len(patches) < num_patches:
            print(f"Not enough valid patches found for the image at {image_path}")
            return None

        patches_tensor = torch.stack(patches[:num_patches])
        return patches_tensor

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def plot_patches(patches_tensor, num_cols=5):
    num_patches = patches_tensor.shape[0]
    num_rows = (num_patches // num_cols) + int(num_patches % num_cols > 0)

    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_patches):
        patch = patches_tensor[i].permute(1, 2, 0).numpy()  # Convert to HWC format for display

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(patch)
        plt.axis('off')

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

if __name__ == '__main__':
    image_path = "00c058_0.tif"
    num_patches = 15

    patches = patch_preprocessing(image_path, num_patches)
    if patches is not None:
        print(patches.shape)
        plot_patches(patches)
