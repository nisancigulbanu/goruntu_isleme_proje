import numpy as np

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    RGB formundaki görüntüyü Luma formülüne (Grayscale) dönüştürür.
    gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 2:
        return image
        
    # Girdi olan görüntüler OpenCV tarafından BGR mantığı ile okunmuş oluyor.
    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    r = image[:, :, 2].astype(float)
    
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

def gray_to_binary(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Gri Pikselleri NumPy maskelemesi kullanarak threshold değerine göre 0 veya 255 yapar.
    """
    if len(image.shape) == 3:
        image = rgb_to_gray(image)
        
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image >= threshold] = 255
    return binary

def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    RGB (veya BGR) matrisini cv2.cvtColor kullanmadan tamamen el ile HSV'ye dönüştürür.
    """
    if len(image.shape) == 2:
        # Görüntü siyah-beyazsa (Gri), RGB'den değil griden işleme düşer
        return image
        
    # pikselleri [0, 1] arasına normalize et
    b = image[:, :, 0] / 255.0
    g = image[:, :, 1] / 255.0
    r = image[:, :, 2] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax  # V = max(R,G,B)

    # Doygunluk Alanı (Saturation)
    mask_v = cmax > 0
    s[mask_v] = delta[mask_v] / cmax[mask_v]

    # Renk Tonu (Hue) hesabı
    # 1. Durum Delta=0 H=0
    # 2. Durum CMAX == R
    mask_r = (cmax == r) & (delta != 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

    # 3. Durum CMAX == G
    mask_g = (cmax == g) & (delta != 0)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)

    # 4. Durum CMAX == B
    mask_b = (cmax == b) & (delta != 0)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Verileri 0-255 uint8 skalasına oturtma (OpenCV HSV Formu H:0-179, S:0-255, V:0-255)
    H_out = (h / 2).astype(np.uint8)
    S_out = (s * 255).astype(np.uint8)
    V_out = (v * 255).astype(np.uint8)

    return np.stack((H_out, S_out, V_out), axis=-1)

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """Verilen formül üzerinden YCbCr'ye (Renk farkı uzayı) dönüştürür."""
    if len(image.shape) == 2:
        return image
        
    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    r = image[:, :, 2].astype(float)
    
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 128.0
    
    y = np.clip(y, 0, 255)
    cb = np.clip(cb, 0, 255)
    cr = np.clip(cr, 0, 255)
    
    return np.stack((y, cb, cr), axis=-1).astype(np.uint8)
