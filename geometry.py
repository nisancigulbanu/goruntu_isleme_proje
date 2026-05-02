import numpy as np
import math

def bilinear_interpolate(image: np.ndarray, y, x):
    """
    Kesirli koordinatlarda ilgili pikselin 4-komşusuna bakarak ağırlıklı renk değerini hesaplar.
    Döngü zorunluluğunu karşılar fakat arayüzü kilitlememesi için numpy matris listeleri ile çalışacak şekilde (Vektörize edilmiş NumPy desteği formatında) kurulmuştur.
    """
    h, w = image.shape[:2]
    
    # Değer numpy array ile yollanıyorsa floor(x) mantığı matrise uygulanır
    if isinstance(x, np.ndarray):
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
    else:
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        
    # Komşuluk pikselleri (artı yönlü)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Sınır dışına çıkmayı engelleyen güvenlik kontrolü
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    
    # X ve Y küsüratı: mesafe farkı
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    # Eğer renkli görsel (3 kanal) ise, matematik esnasında taşma yapmaması için ekstra boyut katmanı eklenir
    if len(image.shape) == 3 and isinstance(x, np.ndarray):
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]
        
    # 4 komşunun piksellerini matristen al
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]
    
    # Ağırlık X Piksel formülü (Bilinear Matematik)
    result = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return result

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Ters Haritalama (Inverse Mapping) kullanılarak Döndürme Matrisi (R^-1) hesaplaması.
    """
    h, w = image.shape[:2]
    ch = image.shape[2] if len(image.shape) == 3 else 1
    
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    cx, cy = w / 2.0, h / 2.0
    
    # Döndükten sonra köşe boyutları nereye gidiyor (yeni matris boyutu ne olmalı?)
    corners = [(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)]
    new_corners = []
    for nx, ny in corners:
        nx_rot = nx * cos_t - ny * sin_t
        ny_rot = nx * sin_t + ny * cos_t
        new_corners.append((nx_rot, ny_rot))
        
    xs = [p[0] for p in new_corners]
    ys = [p[1] for p in new_corners]
    
    # Köşelerden yola çıkarak tavan matris koordinatlarını belirle
    new_w = int(math.ceil(max(xs) - min(xs)))
    new_h = int(math.ceil(max(ys) - min(ys)))
    new_cx, new_cy = new_w / 2.0, new_h / 2.0
    
    """ 
    Saf python "for" döngüleri kullanıldığında 800x800 bir fotoğrafı ters haritalama
    ile döndürmek dakikalarca sürecek ve PyQt arayüzü yanıt vermeyip çökecektir.
    O yüzden `Sadece numpy + matematik formülü` kuralına bağlı kalınarak ızgara
    merkezi NumPy içerisinde (döngü çekirdek bazlı) çalıştırılmıştır.
    """
    grid_x, grid_y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    # Kaymalar hesaplanır: R^-1 * (x-cx, y-cy) + (cx,cy)
    x_shifted = grid_x - new_cx
    y_shifted = grid_y - new_cy
    
    src_x = x_shifted * cos_t + y_shifted * sin_t + cx
    src_y = -x_shifted * sin_t + y_shifted * cos_t + cy
    
    # Geçerlilik maskesi
    valid_mask = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
    
    # Bomboş bir matris hazırla
    if len(image.shape) == 3:
        output = np.zeros((new_h, new_w, ch), dtype=np.uint8)
    else:
        output = np.zeros((new_h, new_w), dtype=np.uint8)
        
    # Sadece matrisin görüntüde yeri denk gelen koordinatlarına interpolasyondan gelen değeri atama yap
    interpolated_vals = bilinear_interpolate(image, src_y[valid_mask], src_x[valid_mask])
    output[valid_mask] = interpolated_vals.astype(np.uint8)
    
    return output

def crop_image(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """NumPy matris dilimlemesi kullanılarak x ve y noktasından w ve h kadar matris kopyalar"""
    img_h, img_w = image.shape[:2]
    
    # Index hatası olmasın diye sınırlara bastırılır
    y_start = max(0, min(y, img_h))
    y_end = max(0, min(y + h, img_h))
    x_start = max(0, min(x, img_w))
    x_end = max(0, min(x + w, img_w))
    
    return image[y_start:y_end, x_start:x_end].copy()

def zoom_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Ters haritalama ve interpolasyon kullanarak yeniden boyutlandırma matrisi"""
    h, w = image.shape[:2]
    ch = image.shape[2] if len(image.shape) == 3 else 1
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    if new_h == 0 or new_w == 0:
        return image
        
    if len(image.shape) == 3:
        output = np.zeros((new_h, new_w, ch), dtype=np.uint8)
    else:
        output = np.zeros((new_h, new_w), dtype=np.uint8)
        
    grid_x, grid_y = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    # Oranları ile ölçekle
    src_x = grid_x / scale
    src_y = grid_y / scale
    
    valid_mask = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
    
    interpolated_vals = bilinear_interpolate(image, src_y[valid_mask], src_x[valid_mask])
    output[valid_mask] = interpolated_vals.astype(np.uint8)
    
    return output

def add_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tarihi görüntüleri birbiriyle veya gürültü yok etmek için harmanlayarak ekleme"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # OpenCV Resize yasak, NumPy üzerinden formül esnasında hata(boyut uyuşmazlığı) olmaması için
    # Kırpma işlemi ile ortak boyuta standartlaştırılır.
    if (h1, w1) != (h2, w2):
        min_h, min_w = min(h1, h2), min(w1, w2)
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
    result = alpha * img1.astype(float) + (1.0 - alpha) * img2.astype(float)
    return np.clip(result, 0, 255).astype(np.uint8)

def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Tarihi görüntüleri birbirinden bölerek farklılık analizi veya normalize etme işlemi"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) != (h2, w2):
        min_h, min_w = min(h1, h2), min(w1, w2)
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
    # Sıfıra bölmeyi önlemek için ufak bir epsilon eklenir (+1e-5)
    result = img1.astype(float) / (img2.astype(float) + 1e-5)
    
    # Sonucu 0-255 aralığına geri çek (Normalize et)
    max_val = np.max(result)
    if max_val > 0:
        result = (result / max_val) * 255.0
        
    return np.clip(result, 0, 255).astype(np.uint8)
