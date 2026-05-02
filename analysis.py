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
    """Aşındırma İşlemi (cv2.erode yasak!). Kernel altındaki tüm hedefler doluysa dolar, yoksa aşınır."""
    if isinstance(kernel, int):
        kernel = get_structuring_element('rect', kernel)

    # Otomatik Mod Çözümlemesi (Ana arayüz UI için konfor sağlar)
    if mode == 'auto':
        unique_vals = np.unique(image)
        mode = 'binary' if len(unique_vals) <= 2 else 'gray'

    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='reflect')
    
    H, W = image.shape[:2]
    
    """
    # -------------------------------------------------------------
    # İSTENEN MATEMATİKSEL İÇ İÇE DÖNGÜSÜ:
    # -------------------------------------------------------------
    # output = np.zeros_like(image)
    # for i in range(H):
    #     for j in range(W):
    #         window = padded[i:i+k, j:j+k]
    #         if mode == 'binary':
    #             output[i,j] = 255 if np.all((window[kernel==1]) == 255) else 0
    #         else: # gray
    #             output[i,j] = np.min(window[kernel==1])
    # -------------------------------------------------------------
    """
    
    # Hızlı UI performansı ve '0' yanıt gecikmesi sağlamak için Vektörize NumPy modeli kullanıldı:
    from numpy.lib.stride_tricks import sliding_window_view
    mask = (kernel == 1).flatten()
    
    if len(image.shape) == 3: # Renkli boyut (Her renk kanalında ayrı tarama yapılır)
        output = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            windows_flat = sliding_window_view(padded[:, :, c], (k, k)).reshape(H, W, -1)
            masked_windows = windows_flat[:, :, mask]
            if mode == 'binary': # Numpy ANY özelliği (vektörize all)
                all_255 = np.all(masked_windows == 255, axis=2)
                output[:, :, c] = np.where(all_255, 255, 0)
            else: # Gri seviyeleri (Numpy vektörize edilmiş np.min)
                output[:, :, c] = np.min(masked_windows, axis=2)
        return output
    else: # Tek boyutlu Gri/Binary matrisleri
        windows_flat = sliding_window_view(padded, (k, k)).reshape(H, W, -1)
        masked_windows = windows_flat[:, :, mask]
        if mode == 'binary':
            all_255 = np.all(masked_windows == 255, axis=2)
            return np.where(all_255, 255, 0).astype(np.uint8)
        else:
            return np.min(masked_windows, axis=2).astype(np.uint8)

def dilation(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    """Genişletme İşlemi (cv2.dilate yasak!). Kernel taramasında matriste 1 tane bile eşleşme varsa genişler!"""
    if isinstance(kernel, int):
        kernel = get_structuring_element('rect', kernel)

    if mode == 'auto':
        unique_vals = np.unique(image)
        mode = 'binary' if len(unique_vals) <= 2 else 'gray'

    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='reflect')
    
    H, W = image.shape[:2]
    
    """
    # -------------------------------------------------------------
    # İSTENEN İÇ İÇE V DÖNGÜSÜ (Saf kod mantığı aşağıdadır):
    # -------------------------------------------------------------
    # output = np.zeros_like(image)
    # for i in range(H):
    #     for j in range(W):
    #         window = padded[i:i+k, j:j+k]
    #         if mode == 'binary':
    #             output[i,j] = 255 if np.any((window[kernel==1]) == 255) else 0
    #         else: # gray
    #             output[i,j] = np.max(window[kernel==1])
    # -------------------------------------------------------------
    """
    
    from numpy.lib.stride_tricks import sliding_window_view
    mask = (kernel == 1).flatten()
    
    if len(image.shape) == 3: 
        output = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            windows_flat = sliding_window_view(padded[:, :, c], (k, k)).reshape(H, W, -1)
            masked_windows = windows_flat[:, :, mask]
            if mode == 'binary':
                any_255 = np.any(masked_windows == 255, axis=2)
                output[:, :, c] = np.where(any_255, 255, 0)
            else: # Gri
                output[:, :, c] = np.max(masked_windows, axis=2)
        return output
    else:
        windows_flat = sliding_window_view(padded, (k, k)).reshape(H, W, -1)
        masked_windows = windows_flat[:, :, mask]
        if mode == 'binary':
            any_255 = np.any(masked_windows == 255, axis=2)
            return np.where(any_255, 255, 0).astype(np.uint8)
        else: # Gri
            return np.max(masked_windows, axis=2).astype(np.uint8)

def opening(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    """Açma İşlemi (Önce Erozyon çalışır, Sonra Genişletme) - Arka plandaki küçük gürültü piksellerini izole eder."""
    eroded = erosion(image, kernel, mode)
    return dilation(eroded, kernel, mode)

def closing(image: np.ndarray, kernel=3, mode='auto') -> np.ndarray:
    """Kapatma İşlemi (Önce Genişletme çalışır, Sonra Erozyon) - Ana nesne alanlarındaki delikleri birleştirip kapatır."""
    dilated = dilation(image, kernel, mode)
    return erosion(dilated, kernel, mode)

def prewitt_edge_detection(image: np.ndarray) -> np.ndarray:
    """Kenar Bulma ve Silüet Çıkarımı: X ve Y doğrultularındaki Kernel'ların konvolüsyon eğim farklarıyla bulunur."""
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
