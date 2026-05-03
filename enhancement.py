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
    """
    Histogramı matplotlib ile çizer. ax=None iken ayrı plt.show() penceresi açar; PyQt gömülü arayüzde
    bu yol olay döngüsüyle çakışıp çökme riski taşıyabilir — tercihen MainWindow.show_histogram_dialog kullanın.
    """
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
    # Elif: Kontrastı artırmak için pikselleri P_yeni = (P_eski - min) / (max - min) * 255 formülüyle 
    # tüm matrise manuel yayıyoruz. İç içe for döngüsü kullandık.
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    
    if len(image.shape) == 3:
        ch = image.shape[2]
        for c in range(ch):
            channel = image[:,:,c]
            mn = int(np.min(channel))
            mx = int(np.max(channel))
            diff = mx - mn
            if diff == 0:
                diff = 1
            # Düşük seviyeli piksel döngüleriyle manuel yayma
            for y in range(h):
                for x in range(w):
                    val = image[y, x, c]
                    new_val = int((val - mn) / diff * 255)
                    out[y, x, c] = new_val
    else:
        mn = int(np.min(image))
        mx = int(np.max(image))
        diff = mx - mn
        if diff == 0:
            diff = 1
        for y in range(h):
            for x in range(w):
                val = image[y, x]
                new_val = int((val - mn) / diff * 255)
                out[y, x] = new_val
                
    return out

def contrast_enhancement(image: np.ndarray, factor: float) -> np.ndarray:
    # Öğrenci notu: Alfa çarpımı yerine 128 (orta nokta) baz alınarak lineer kontrast artırdık.
    result = 128.0 + factor * (image.astype(np.float64) - 128.0)
    return np.clip(result, 0, 255).astype(np.uint8)

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """İnsan gözünün logaritmik algısına uydurmak için veya karanlık restorasyonu için Gamma düzeltmesi"""
    normalized = image.astype(np.float64) / 255.0
    corrected = (normalized ** gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    # Öğrenci notu: Hata verisi testi için resme rastgele siyah (0) ve beyaz (255) pikseller serpiştiriyoruz.
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
    main.py uyumu: ayrı argümanlar (img1, img2), liste [img1, img2] veya tek tuple (img1, img2) kabul edilir.
    Boyutlar (yükseklik × genişlik × kanal) birebir aynı değilse ValueError fırlatılır (sessiz kırpma yok).
    """
    if len(args) == 1:
        first = args[0]
        if isinstance(first, np.ndarray):
            images_list = [first]
        elif isinstance(first, (list, tuple)):
            images_list = list(first)
        else:
            raise TypeError("Görüntü listesi NumPy ndarray, liste veya demet olmalıdır.")
    else:
        images_list = list(args)

    if not images_list:
        raise ValueError("Ortalama almak için en az 1 görüntü gereklidir.")
    if not all(isinstance(im, np.ndarray) for im in images_list):
        raise TypeError("Tüm öğeler NumPy ndarray olmalıdır.")

    ref_shape = images_list[0].shape
    for idx, img in enumerate(images_list[1:], start=2):
        if img.shape != ref_shape:
            raise ValueError(
                f"Görüntü boyutları eşleşmiyor (1. görüntü {ref_shape}, {idx}. görüntü {img.shape}). "
                "Ortalama için tüm görüntülerin yükseklik, genişlik ve kanal sayısı aynı olmalıdır."
            )

    stack = np.array(images_list, dtype=np.float64)
    avg_img = np.mean(stack, axis=0)
    return np.clip(avg_img, 0, 255).astype(np.uint8)
