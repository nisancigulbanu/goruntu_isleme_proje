import numpy as np

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tüm uzamsal filtrelerin konvolüsyon işlemi için kullandığı CORE (ÇEKİRDEK) FONKSİYON."""
    if len(image.shape) == 3: # Renkli Görüntü
        output = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            output[:, :, c] = _convolve2d_single(image[:, :, c], kernel)
        return output.astype(np.uint8)
    else: # Gri/Binary Görüntü
        return _convolve2d_single(image, kernel).astype(np.uint8)

def _convolve2d_single(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = image.shape
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros((H, W), dtype=np.float64)

    # Bengü: Kernel matrisini resmin pikselleri üzerinde gezdirerek konvolüsyon işlemi yapıyoruz (sliding window).
    for i in range(H):
        for j in range(W):
            # Pencereyi alıp kernel ile eleman bazlı çarpıp topluyoruz
            window = padded[i:i+k, j:j+k]
            output[i, j] = np.sum(window * kernel)

    return np.clip(output, 0, 255)

def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Mean (Ortalama/Blur) Filtresi - İçerisindeki her ağırlık eşittir."""
    k = int(kernel_size)
    if k < 1:
        k = 3
    if k % 2 == 0:
        k += 1
    h, w = image.shape[:2]
    if h < k or w < k:
        raise ValueError(
            f"Görüntü ({w}×{h}) çekirdek boyutundan ({k}) küçük; önce daha büyük görüntü yükleyin veya kırpın."
        )
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return convolve2d(image, kernel)

def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # Median Filtresi: Konvolüsyonla çarpma yok, pikselleri sıralayıp medyan değerini elle buluyoruz. (cv2.medianBlur yasak!)
    k = int(kernel_size)
    if k < 1:
        k = 3
    if k % 2 == 0:
        k += 1
    h, w = image.shape[:2]
    if h < k or w < k:
        raise ValueError(
            f"Görüntü ({w}×{h}) pencereden ({k}×{k}) küçük; median uygulanamaz."
        )
    kernel_size = k
    if len(image.shape) == 3:
        output = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            output[:, :, c] = _median_single(image[:, :, c], kernel_size)
        return output
    else:
        return _median_single(image, kernel_size)

def _median_single(image: np.ndarray, kernel_size: int) -> np.ndarray:
    H, W = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros((H, W), dtype=np.uint8)
    
    # Elif: Median filtresini sıralama algoritmasıyla manuel yazdık, matris üzerinde kernel'i kaydırıp
    # pikselleri küçükten büyüğe sıralayarak (sort) ortadaki medyan değeri seçiyoruz.
    for i in range(H):
        for j in range(W):
            window = padded[i:i+kernel_size, j:j+kernel_size].flatten()
            sorted_pixels = np.sort(window)
            output[i, j] = sorted_pixels[len(sorted_pixels) // 2]
            
    return output

def manuel_gaussian_kernel(sigma: float) -> np.ndarray:
    """Unsharp Masking (Keskinleştirme) işlevi için kullanılacak olan Gauss Çekirdeği Matematiği"""
    size = int(6.0 * sigma)
    if size % 2 == 0:
        size += 1 # Matris penceresi kuralı gereğince tek sayı olmalı (örn: 3x3 merkez)
    if size < 3: size = 3
        
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    
    # Gauss Fonksiyon Formülü -> e ^ -( (x^2 + y^2) / 2*sigma^2 )
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # Oranları 1.0 üzerinden normalize et 
    return g / g.sum()

def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    # Gülbanu: Önce manuel konvolüsyonla blur alıyoruz, orijinalden çıkarıp maskeyi (kenarları) buluyoruz.
    # Sonra bu maskeyi orijinal resme ekleyerek keskinleştiriyoruz. Hazır sharpen fonksiyonu kullanmadık.
    
    # GUI (Arayüz) Uyumluluk Köprüsü: main.py parametre olarak tek slider (amount) yolladıysa,
    # bu 2. argüman olan sigma'ya yansır. İşin formülize kısmında amount(0.5 - 3.0) aslında `strength`'tir.
    if sigma > 1.0 and strength == 1.5: 
        strength = sigma
        sigma = 1.0 # Ortalama Blur tutulur
        
    # Gaussian kernel MANUEL oluştur
    gaussian_kernel = manuel_gaussian_kernel(sigma)
    
    # 1. Konvolüsyonla blur al (Low-pass)
    blurred = convolve2d(image, gaussian_kernel)
    
    # 2. Mantıksal çıkarım: Maskeyi bul (Edge / Detay)
    img_float = image.astype(np.float64)
    mask = img_float - blurred.astype(np.float64)
    
    # 3. Orijinale maskeyi belirlenen çarpım kuvvetiyle (strength) iade et
    sharpened = img_float + strength * mask
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)
