"""
preprocessing
=============

Preprocesamiento de imagenes de nanohilos previo a la extraccion del esqueleto
(``SkeletonAnalysis``). Portado desde las celdas sueltas de ``temp.py``.
"""

import cv2
import numpy as np


def load_gray(path):
    """Carga una imagen en escala de grises (uint8)."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55):
    """
    Binarizacion adaptativa gaussiana.

    Aplica un umbral adaptativo (media gaussiana local por bloques) con padding
    replicado para evitar artefactos de borde, y ademas fuerza a fondo (0) los
    pixeles por debajo de ``intensity_thresh``.

    Parametros
    ----------
    img : np.ndarray
        Imagen en escala de grises (uint8).
    block_size : int (impar)
        Tamano de la ventana local del umbral adaptativo.
    C : int
        Constante que se resta a la media local.
    intensity_thresh : int
        Intensidad minima; por debajo se fuerza a fondo.

    Devuelve
    --------
    np.ndarray : mascara binaria (0 / 255).
    """
    if block_size % 2 == 0:
        raise ValueError("block_size debe ser impar.")

    pad = block_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    mask_padded = cv2.adaptiveThreshold(
        img_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    mask = mask_padded[pad:-pad, pad:-pad]
    mask[img < intensity_thresh] = 0
    return mask


def overlay_mask(base_gray, binary_mask, color_rgb=(220, 20, 60)):
    """
    Superpone en color una mascara binaria sobre una imagen en gris.
    Se consideran 'objeto' los pixeles con ``mask < 127``.

    Devuelve una imagen RGB (np.ndarray) para visualizar el resultado de la
    binarizacion sobre la imagen original.
    """
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    base_rgb[binary_mask < 127] = color_rgb
    return base_rgb
