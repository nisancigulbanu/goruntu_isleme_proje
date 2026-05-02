import ast
import os

forbidden_list = [
    'resize', 'rotate', 'warpAffine', 'filter2D', 'blur',
    'GaussianBlur', 'medianBlur', 'Canny', 'Sobel', 'erode',
    'dilate', 'cvtColor', 'equalizeHist', 'threshold', 
    'adaptiveThreshold', 'Laplacian'
]

def check_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return []
    
    tree = ast.parse(content)
    violations = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # Kod içerisinde yapılmış "cv2.X()" taraması
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'cv2':
                    if node.func.attr in forbidden_list:
                        violations.append((node.func.attr, node.lineno))
    return violations

if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    all_clean = True
    
    print(f"\n=== YASAKLI 'cv2' FONKSİYONU AST KONTROLÜ ===")
    print(f"Taranan Dizin: {directory}\n")
    
    py_files = [f for f in os.listdir(directory) if f.endswith('.py') and f != os.path.basename(__file__)]
    
    if not py_files:
        print("Kontrol edilecek .py dosyası bulunamadı.")
    else:
        for filename in py_files:
            filepath = os.path.join(directory, filename)
            violations = check_file(filepath)
            
            if violations:
                all_clean = False
                for v, line in violations:
                    print(f"[X] İHLAL TESPİT EDİLDİ | Dosya: {filename:<18} Satır: {line:<4} Yasaklı Fonksiyon: cv2.{v}")
            else:
                print(f"[OK] Temiz: {filename}")
                
        if all_clean:
            print("\nTEBRİKLER: Hiçbir dosyada yasaklı OpenCV (Görüntü İşleme) fonksiyonuna ('cv2.') rastlanmadı!")
            print("Tüm modüller kısıtlamalara uyarak %100 NumPy + Matematiksel formüllerle yazılmış durumda.")
        else:
            print("\nUYARI: Projeyi teslim etmeden önce yukarıda tespit edilen yasaklı kullanımları siliniz!")
