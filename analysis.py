import numpy as np
from filtering import convolve2d

def get_structuring_element(shape: str, size: int) -> np.ndarray:
    """Belirtilen şekilde ('rect', 'cross', 'ellipse') binary (0-1) çekirdek(kernel) maskesi üretir."""
    if shape == 'rect':
        return np.ones((size, size), dtype=np.uint8)
    elif shape == 'cross':
        kernel = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        kernel[mid, :] = 1
        kernel[:, mid] = 1
        return kernel
    elif shape == 'ellipse':
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                # Elips/Daire alan hesabı
                if ((i - center)**2 + (j - center)**2) <= (center**2):
                    kernel[i, j] = 1
        return kernel
    else:
        return np.ones((size, size), dtype=np.uint8)

def erosion(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    # Gülbanu: Aşındırma (Erosion) işlemi için kernel matrisini resimde döngüyle kaydırarak uyguluyoruz.
    if isinstance(kernel, int):
        kernel = get_structuring_element('rect', kernel)

    if mode == 'auto':
        unique_vals = np.unique(image)
        mode = 'binary' if len(unique_vals) <= 2 else 'gray'

    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='reflect')
    
    H, W = image.shape[:2]
    
    if len(image.shape) == 3:
        output = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            for i in range(H):
                for j in range(W):
                    window = padded[i:i+k, j:j+k, c]
                    if mode == 'binary':
                        output[i, j, c] = 255 if np.all(window[kernel == 1] == 255) else 0
                    else:
                        output[i, j, c] = np.min(window[kernel == 1])
        return output
    else:
        output = np.zeros_like(image, dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                window = padded[i:i+k, j:j+k]
                if mode == 'binary':
                    output[i, j] = 255 if np.all(window[kernel == 1] == 255) else 0
                else:
                    output[i, j] = np.min(window[kernel == 1])
        return output

def dilation(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    # Nazlı: Genişletme (Dilation) yaparken kernel'i piksellerin üzerinde gezdiriyoruz, 
    # matriste maskeyle eşleşen 1 tane bile piksel varsa o alanı genişletiyoruz.
    if isinstance(kernel, int):
        kernel = get_structuring_element('rect', kernel)

    if mode == 'auto':
        unique_vals = np.unique(image)
        mode = 'binary' if len(unique_vals) <= 2 else 'gray'

    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='reflect')
    
    H, W = image.shape[:2]
    
    if len(image.shape) == 3: 
        output = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            for i in range(H):
                for j in range(W):
                    window = padded[i:i+k, j:j+k, c]
                    if mode == 'binary':
                        output[i, j, c] = 255 if np.any(window[kernel == 1] == 255) else 0
                    else:
                        output[i, j, c] = np.max(window[kernel == 1])
        return output
    else:
        output = np.zeros_like(image, dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                window = padded[i:i+k, j:j+k]
                if mode == 'binary':
                    output[i, j] = 255 if np.any(window[kernel == 1] == 255) else 0
                else:
                    output[i, j] = np.max(window[kernel == 1])
        return output

def opening(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    """Açma İşlemi (Önce Erozyon çalışır, Sonra Genişletme) - Arka plandaki küçük gürültü piksellerini izole eder."""
    eroded = erosion(image, kernel, mode)
    return dilation(eroded, kernel, mode)

def closing(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    """Kapatma İşlemi (Önce Genişletme çalışır, Sonra Erozyon) - Ana nesne alanlarındaki delikleri birleştirip kapatır."""
    dilated = dilation(image, kernel, mode)
    return erosion(dilated, kernel, mode)

def prewitt_edge_detection(image: np.ndarray) -> np.ndarray:
    # Bengü: Burada Prewitt kernelini resmin üstünde döngüyle gezdirerek X ve Y türevlerini manuel hesaplıyoruz.
    Kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    Kernel_y = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float64)
    
    # Gradyanların tespiti için sadece parlaklık üzerinden (Luma Gray) çalıştırılır
    if len(image.shape) == 3:
        process_img = (0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]).astype(np.uint8)
    else:
        process_img = image
        
    # X ve Y eksenli Convolution çalıştır (filtering module)
    Gx = convolve2d(process_img, Kernel_x).astype(np.float64)
    Gy = convolve2d(process_img, Kernel_y).astype(np.float64)
    
    # Birleşik Türev Büyüklüğü (Magnitude Formülü)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    return np.clip(magnitude, 0, 255).astype(np.uint8)
