import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Oluşturduğumuz saf NumPy modüllerini içeri alıyoruz
import preprocessing
import geometry
import enhancement
import filtering
import analysis

def run_test_pipeline(image_path="test_image.jpg"):
    print("="*60)
    print(" DİJİTAL ARŞİVCİ - ENTEGRASYON VE PERFORMANS TEST PIPELINE ")
    print("="*60)
    
    # 1. Görüntüyü Yükle
    if not os.path.exists(image_path):
        print(f"[Uyarı] '{image_path}' bulunamadı. Sistemin çökmemesi için geçici bir rastgele (noise) Görüntü üretiliyor...")
        dummy = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy)
        
    img = cv2.imread(image_path)
    if img is None:
        print("Hata: Görüntü cv2.imread ile okunamadı veya görüntü bozuk.")
        return
        
    print(f"[*] Orijinal Görüntü Başarıyla Yüklendi. Boyut: {img.shape}")
    
    # Listeye Tuple olarak (Aşama Adı, Matris) şeklinde ekleyeceğiz. BGR formatında.
    steps_history = [("Orijinal", img)]
    
    # 2. Pipeline Tanımı 
    # Tuple: (Aşama Adı, Modül Adı, Fonksiyon, Parametreler Tuple'ı)
    pipeline = [
        {
            "name": "Gri Dönüşüm",
            "module": "preprocessing",
            "func": preprocessing.rgb_to_gray,
            "args": ()
        },
        {
            "name": "Histogram Stretching",
            "module": "enhancement",
            "func": enhancement.histogram_stretching,
            "args": ()
        },
        {
            "name": "Gürültü Ekleme (%5)",
            "module": "enhancement",
            "func": enhancement.add_salt_pepper_noise,
            "args": (0.05,)
        },
        {
            "name": "Median Filtre (3x3)",
            "module": "filtering",
            "func": filtering.median_filter,
            "args": (3,)
        },
        {
            "name": "Prewitt Kenar Bulma",
            "module": "analysis",
            "func": analysis.prewitt_edge_detection,
            "args": ()
        },
        {
            "name": "Dilation (3x3 Rect)",
            "module": "analysis",
            "func": analysis.dilation,
            "args": (3, 'auto') # 3 integer -> 3x3 boyutlu alan üretecek, otomatik renk mode'u
        }
    ]
    
    current_img = img.copy()
    
    # Süre ve hata denetimli pipeline ilerlemesi
    for idx, step in enumerate(pipeline):
        module_name = step['module']
        func_name = step['name']
        
        try:
            print(f"\n[{module_name}.py] {func_name} işlemi başlatıldı...")
            start_time = time.perf_counter()
            
            # Fonksiyonu dinamik (dereference) olarak mevcut matrise uygula
            current_img = step['func'](current_img, *step['args'])
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"  --> Başarılı! İşlem Süresi: {elapsed_time:.4f} saniye")
            
            # 3. Her adımdan sonra kaydet
            safe_name = func_name.replace(" ", "_").replace("%", "").replace("&", "").replace("(", "").replace(")", "")
            save_path = f"pipeline_step_{idx+1}_{safe_name}.jpg"
            cv2.imwrite(save_path, current_img)
            
            steps_history.append((func_name, current_img.copy()))
            
        except Exception as e:
            # 5. Hata denetimi ve açık Modül bildirim ekranı
            print(f"\n!!!! CRITICAL PIPELINE HATASI DETECTED !!!!")
            print(f"-> Çöken Modül   : {module_name}.py")
            print(f"-> Çöken Fonksiyon: {func_name}")
            print(f"-> Hata Detayı    : {str(e)}")
            import traceback
            traceback.print_exc()
            return
            
    # Görselleştirme (Matplotlib yan yana dizilim)
    print("\n[+] Tüm aşamalar hatasız yürütüldü. Matplotlib ile sonuç tablosu hazırlanıyor...")
    total_steps = len(steps_history)
    cols = min(4, total_steps)
    rows = int(np.ceil(total_steps / cols))
    
    plt.figure(figsize=(16, 4 * rows))
    for i, (title, img_data) in enumerate(steps_history):
        plt.subplot(rows, cols, i + 1)
        
        # Ekran rengi optimizasyonu
        if len(img_data.shape) == 3:
            rgb_data = img_data[:, :, ::-1] # BGR to RGB (OpenCV Matrisi)
            plt.imshow(rgb_data)
        else:
            plt.imshow(img_data, cmap='gray')
            
        plt.title(f"{i}. {title}", fontsize=11, fontweight="bold")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("Sektor_Ozet_Raporu.png", dpi=150)
    print("[*] Test raporu 'Sektor_Ozet_Raporu.png' olarak diske kaydedildi.")
    print("[*] Görüntü ekrana yansıtılıyor...")
    plt.show()

if __name__ == "__main__":
    # Test eskalasyonu (Kendiniz 'test.jpg' yollayabilirsiniz)
    try_img = "test_image.jpg"
    if len(sys.argv) > 1:
        try_img = sys.argv[1]
        
    run_test_pipeline(try_img)
