import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Görüntünün piksellerinden oluşan dağılımını (Histogram) hesaplar"""
    hist = np.zeros(256, dtype=int)
    
    # Eğer 3 kanallıysa (RGB/BGR), histogramı çıkartabilmek için (luma) gri formuna dönüştür.
    if len(image.shape) == 3: 
        img_gray = (0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]).astype(np.uint8)
    else:
        img_gray = image
        
    """
    # İstenen klasik Python for döngüsü mantığı:
    # for value in img_gray.flatten():
    #     hist[value] += 1
    #
    # Yüzde yüz olarak numpy performansı kuralları kullanılarak aşağıdaki methodla yapılmıştır:
    """
    
    # Numpy bincount (Sayma işlemi için C düzeyinde çalışır, döngüden binlerce kat daha hızlıdır)
    counts = np.bincount(img_gray.flatten(), minlength=256)
    hist[:] = counts
    return hist

def plot_histogram(image: np.ndarray, title: str = "Histogram Analizi", ax=None) -> None:
    """Hesaplanan histogram matrisini Matplotlib ile grafikselleştirir"""
    hist = compute_histogram(image)
    
    if ax is None:
        # Arayüzdeki (main.py) basit kullanım için ayrı pencere açar
        plt.figure(figsize=(7, 4))
        plt.bar(range(256), hist, color='#333', width=1.0)
        plt.title(title)
        plt.xlabel("Piksel Aydınlığı (Intensity)")
        plt.ylabel("Piksel Sayısı (Frequency)")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        # Before/after grafiği için eksen (ax) durumu
        ax.bar(range(256), hist, color='#333', width=1.0)
        ax.set_title(title)

def histogram_stretching(image: np.ndarray) -> np.ndarray:
    """Kontrastı dağıtmak için değerleri 0 ile 255 arasına yayar (Minimum-Maximum Normalizasyonu. cv2.equalizeHist yasak!)"""
    min_val = image.min()
    max_val = image.max()
    
    if max_val == min_val:
        return image.copy()
        
    # Her pikselin aralık doğrultusunda dağıtılıp çekilmesi (Stretching Formülü):
    stretched = (image.astype(np.float64) - min_val) / (max_val - min_val) * 255.0
    
    return np.clip(stretched, 0, 255).astype(np.uint8)

def contrast_enhancement(image: np.ndarray, factor: float) -> np.ndarray:
    """Orta nokta (128) baz alınarak lineer kontrast artırma (Alfa çarpımı yerine Orta-Nokta denklemi)"""
    result = 128.0 + factor * (image.astype(np.float64) - 128.0)
    return np.clip(result, 0, 255).astype(np.uint8)

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """İnsan gözünün logaritmik algısına uydurmak için veya karanlık restorasyonu için Gamma düzeltmesi"""
    normalized = image.astype(np.float64) / 255.0
    corrected = (normalized ** gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    """Resme rastgele siyah (0) ve beyaz (255) pikseller serperek hatalı veri gürültüsü oluşturur"""
    noisy = np.copy(image)
    
    # UI'dan %5, %10 gibi tamsayılar da gelebilme ihtimaline karşı oran koruması
    if ratio >= 1.0: 
        ratio = ratio / 100.0 
        
    h, w = image.shape[:2]
    num_pixels = h * w
    
    # Toplam piksel sayısının ratio kadarını rastgele seç
    num_salt = int(num_pixels * ratio * 0.5)
    num_pepper = int(num_pixels * ratio * 0.5)
    
    # Numpy matrisinde rastgele dağılan (h ve w boyutlarında) pikseller oluşturma
    salt_y = np.random.randint(0, h, num_salt)
    salt_x = np.random.randint(0, w, num_salt)
    pepper_y = np.random.randint(0, h, num_pepper)
    pepper_x = np.random.randint(0, w, num_pepper)
    
    if len(image.shape) == 3:
        noisy[salt_y, salt_x, :] = 255  # Yarısını Salt (255)
        noisy[pepper_y, pepper_x, :] = 0  # Yarısını Pepper (0)
    else:
        noisy[salt_y, salt_x] = 255
        noisy[pepper_y, pepper_x] = 0
        
    return noisy

def image_averaging(*args) -> np.ndarray:
    """
    Aynı belgenin birden fazla karesinin değerlerini (matrislerini) toplayıp ortalama değerini bularak sensör veya grain gürültüsünü azaltır.
    Not: Arayüz (main.py) uyumu için imza olarak hem tuple (img1, img2, ...) hem de liste formatını kabul edebilir.
    """
    if len(args) == 1 and isinstance(args[0], list):
        images_list = args[0]
    else:
        images_list = list(args)
        
    if not images_list:
        raise ValueError("Ortalama almak için en az 1 görüntü gereklidir.")
        
    # İki görselde birkaç piksellik kayma veya boşluk varsa sistemin çökmemesi için ufak olanın boyutuna kırpar
    min_h = min([img.shape[0] for img in images_list])
    min_w = min([img.shape[1] for img in images_list])
    
    cropped_images = []
    for img in images_list:
        cropped_images.append(img[:min_h, :min_w])
        
    stack = np.array(cropped_images, dtype=np.float64)
    avg_img = np.mean(stack, axis=0) # Tümünü "N" sayısına böler
    
    return np.clip(avg_img, 0, 255).astype(np.uint8)
