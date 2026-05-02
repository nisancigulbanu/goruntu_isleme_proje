import sys
import os
import html
import cv2
import numpy as np

def safe_imread(path):
    try:
        # OpenCV imread Türkçe karakterli yollarda None döndürür, bu yüzden np.fromfile + cv2.imdecode kullanıyoruz.
        img_array = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def safe_imwrite(path, img):
    try:
        # Türkçe karakterleri desteklemesi için imencode ile yazıyoruz
        ext = os.path.splitext(path)[1]
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(path)
            return True
        return False
    except Exception:
        return False

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTabWidget, QSlider, QSpinBox,
    QMessageBox, QFileDialog, QToolBar, QAction, QListWidget,
    QComboBox, QGroupBox, QSizePolicy, QDialog, QCheckBox, QDoubleSpinBox,
    QFrame, QShortcut,
)

# Gerçek modüller ilerleyen aşamalarda kodlanana kadar, PyQt arayüzünün çökmemesi
# ve butonların test edilebilmesi için sahte (dummy) modül yapısı hazırlıyoruz.
class DummyModule:
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"'{name}' fonksiyonu henüz kodlanmadı. Sonraki aşamalarda (NumPy ile) eklenecek.")
        return wrapper

try:
    import preprocessing
except ImportError:
    preprocessing = DummyModule()

try:
    import geometry
except ImportError:
    geometry = DummyModule()

try:
    import enhancement
except ImportError:
    enhancement = DummyModule()

try:
    import filtering
except ImportError:
    filtering = DummyModule()

try:
    import analysis
except ImportError:
    analysis = DummyModule()


class BatchProcessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Toplu İşlem (Batch) Yöneticisi")
        self.resize(500, 550)
        
        self.layout = QVBoxLayout(self)
        
        self.group_ops = QGroupBox("Uygulanacak Filtreleri Seçin ve Ayarlayın (Sırasıyla Uygulanır)")
        self.vbox_ops = QVBoxLayout()
        
        # 1. Gri Dönüşüm
        self.chk_gray = QCheckBox("Gri Dönüşüm (Grayscale)")
        self.vbox_ops.addWidget(self.chk_gray)
        
        # 2. Binary Dönüşüm
        h_bin = QHBoxLayout()
        self.chk_binary = QCheckBox("Binary Dönüşüm")
        self.spin_binary = QSpinBox()
        self.spin_binary.setRange(0, 255)
        self.spin_binary.setValue(128)
        h_bin.addWidget(self.chk_binary)
        h_bin.addWidget(QLabel("Eşik (0-255):"))
        h_bin.addWidget(self.spin_binary)
        h_bin.addStretch()
        self.vbox_ops.addLayout(h_bin)
        
        # 3. Histogram Germe
        self.chk_hist = QCheckBox("Histogram Germe (Stretching)")
        self.vbox_ops.addWidget(self.chk_hist)
        
        # 4. Kontrast Artır
        h_cont = QHBoxLayout()
        self.chk_contrast = QCheckBox("Kontrast Artır")
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.1, 5.0)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setValue(1.5)
        h_cont.addWidget(self.chk_contrast)
        h_cont.addWidget(QLabel("Çarpan (0.1x - 5.0x):"))
        h_cont.addWidget(self.spin_contrast)
        h_cont.addStretch()
        self.vbox_ops.addLayout(h_cont)
        
        # 5. Median Filtresi
        h_med = QHBoxLayout()
        self.chk_median = QCheckBox("Gürültü Temizle (Median)")
        self.combo_median = QComboBox()
        self.combo_median.addItems(["3", "5", "7", "9"])
        h_med.addWidget(self.chk_median)
        h_med.addWidget(QLabel("Kernel Boyutu (KxK):"))
        h_med.addWidget(self.combo_median)
        h_med.addStretch()
        self.vbox_ops.addLayout(h_med)
        
        # 6. Keskinleştirme
        h_unsharp = QHBoxLayout()
        self.chk_unsharp = QCheckBox("Keskinleştirme (Unsharp Masking)")
        self.spin_unsharp = QDoubleSpinBox()
        self.spin_unsharp.setRange(0.1, 5.0)
        self.spin_unsharp.setSingleStep(0.1)
        self.spin_unsharp.setValue(1.5)
        h_unsharp.addWidget(self.chk_unsharp)
        h_unsharp.addWidget(QLabel("Güç/Strength:"))
        h_unsharp.addWidget(self.spin_unsharp)
        h_unsharp.addStretch()
        self.vbox_ops.addLayout(h_unsharp)
        
        # 7. Kenar Çıkarma
        self.chk_prewitt = QCheckBox("Kenar Çıkarma (Prewitt Edge)")
        self.vbox_ops.addWidget(self.chk_prewitt)
        
        self.group_ops.setLayout(self.vbox_ops)
        self.layout.addWidget(self.group_ops)
        
        self.input_files = []
        self.output_folder = ""
        
        self.btn_in = QPushButton("1) İşlenecek Görüntüleri Seç (Birden fazla seçebilirsiniz)")
        self.btn_in.clicked.connect(self.select_in)
        self.lbl_in = QLabel("Görüntü seçilmedi")
        
        self.btn_out = QPushButton("2) Çıktı Klasörü Seç (Kaydedilecek Klasör)")
        self.btn_out.clicked.connect(self.select_out)
        self.lbl_out = QLabel("Seçilmedi")
        
        self.layout.addWidget(self.btn_in)
        self.layout.addWidget(self.lbl_in)
        self.layout.addWidget(self.btn_out)
        self.layout.addWidget(self.lbl_out)
        
        self.btn_start = QPushButton("Toplu İşlemi Başlat")
        self.btn_start.clicked.connect(self.accept)
        self.btn_start.setStyleSheet("background-color: #4a90e2; color: white; font-weight: bold; margin-top: 10px; padding: 10px;")
        self.layout.addWidget(self.btn_start)
        
    def select_in(self):
        files, _ = QFileDialog.getOpenFileNames(self, "İşlenecek Görüntüleri Seç", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if files:
            self.input_files = files
            self.lbl_in.setText(f"{len(files)} adet görüntü seçildi")
            
    def select_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı Klasörü Seç")
        if folder:
            self.output_folder = folder
            self.lbl_out.setText(folder)
            
    def get_operations(self):
        ops = []
        if self.chk_gray.isChecked(): ops.append((preprocessing.rgb_to_gray, []))
        if self.chk_binary.isChecked(): ops.append((preprocessing.gray_to_binary, [self.spin_binary.value()]))
        if self.chk_hist.isChecked(): ops.append((enhancement.histogram_stretching, []))
        if self.chk_contrast.isChecked(): ops.append((enhancement.contrast_enhancement, [self.spin_contrast.value()]))
        if self.chk_median.isChecked(): ops.append((filtering.median_filter, [int(self.combo_median.currentText())]))
        if self.chk_unsharp.isChecked(): ops.append((filtering.unsharp_mask, [self.spin_unsharp.value()]))
        if self.chk_prewitt.isChecked(): ops.append((analysis.prewitt_edge_detection, []))
        return ops

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijital Arşivci - Tarihi Belge ve Fotoğraf Restorasyonu")
        self.resize(1150, 820)
        self.setMinimumSize(860, 620)

        self.original_image = None
        self.current_image = None
        self.image_path = None
        # history yapısı: (islem_adi, fonksiyon, parametreler, onceki_durum_matrisi)
        self.history = []

        self.setup_ui()

        quit_shortcut = QShortcut(QKeySequence(Qt.Key_Q), self)
        quit_shortcut.setContext(Qt.WindowShortcut)
        quit_shortcut.activated.connect(self.close)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 6, 8, 8)

        top_h_layout = QHBoxLayout()

        left_col = QVBoxLayout()
        self.lbl_orig_meta = QLabel()
        self.lbl_orig_meta.setObjectName("panelMeta")
        self.lbl_orig_meta.setTextFormat(Qt.RichText)
        self.lbl_orig_meta.setWordWrap(True)
        self.lbl_orig_meta.setMinimumHeight(34)
        self.left_label = QLabel(
            "<div align='center' style='line-height:1.55'>"
            "<span style='font-size:15px;font-weight:600;color:#9aa3b8'>Kaynak görüntü</span><br/>"
            "<span style='font-size:12px;color:#5f677a'>Yüklemek için araç çubuğundan « Görüntü Yükle » kullanın</span>"
            "</div>"
        )
        self.left_label.setTextFormat(Qt.RichText)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setObjectName("panelImage")
        self.left_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_label.setMinimumSize(120, 80)
        left_col.addWidget(self.lbl_orig_meta)
        left_col.addWidget(self.left_label, stretch=1)
        left_wrap = QWidget()
        left_wrap.setLayout(left_col)
        top_h_layout.addWidget(left_wrap, stretch=2)

        right_col = QVBoxLayout()
        self.lbl_proc_meta = QLabel()
        self.lbl_proc_meta.setObjectName("panelMeta")
        self.lbl_proc_meta.setTextFormat(Qt.RichText)
        self.lbl_proc_meta.setWordWrap(True)
        self.lbl_proc_meta.setMinimumHeight(34)
        self.right_label = QLabel(
            "<div align='center' style='line-height:1.55'>"
            "<span style='font-size:15px;font-weight:600;color:#9aa3b8'>İşlenmiş önizleme</span><br/>"
            "<span style='font-size:12px;color:#5f677a'>İşlemler bu panelde görünür — sekme menüsünden araç seçin</span>"
            "</div>"
        )
        self.right_label.setTextFormat(Qt.RichText)
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setObjectName("panelImage")
        self.right_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_label.setMinimumSize(120, 80)
        right_col.addWidget(self.lbl_proc_meta)
        right_col.addWidget(self.right_label, stretch=1)
        right_wrap = QWidget()
        right_wrap.setLayout(right_col)
        top_h_layout.addWidget(right_wrap, stretch=2)

        history_group = QGroupBox("İşlem Geçmişi")
        history_group.setObjectName("historyCard")
        history_layout = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.setToolTip(
            "Her satır numaralıdır ve parametreleri içerir. Bir adıma dönmek için satıra çift tıklayın."
        )
        self.history_list.itemDoubleClicked.connect(self.undo_operation_from_list)
        history_layout.addWidget(self.history_list)

        h_hist_btns = QHBoxLayout()
        self.btn_undo = QPushButton("Son İşlemi Geri Al (Undo)")
        self.btn_undo.setToolTip("Yalnızca son uygulanan işlemi geri alır.")
        self.btn_undo.clicked.connect(self.undo_operation)
        self.btn_clear_history = QPushButton("Geçmişi Temizle")
        self.btn_clear_history.setToolTip(
            "Geçmiş listesini ve geri al zincirini temizler; görüntünün şu anki hâli korunur."
        )
        self.btn_clear_history.clicked.connect(self.clear_history)
        h_hist_btns.addWidget(self.btn_undo)
        h_hist_btns.addWidget(self.btn_clear_history)
        history_layout.addLayout(h_hist_btns)

        history_group.setLayout(history_layout)
        top_h_layout.addWidget(history_group, stretch=1)

        # Tüm dikey büyüme önizleme + geçmişe gider; sekme bandı sabit yükseklikte kalır.
        main_layout.addLayout(top_h_layout, stretch=1)

        sep = QFrame()
        sep.setObjectName("tabDivider")
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setFixedHeight(2)
        sep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(sep)

        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(175)
        self.tabs.setMaximumHeight(328)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        main_layout.addWidget(self.tabs, stretch=0)

        self.setup_tabs()
        self.setup_toolbar()
        self.update_panel_metadata()

    def setup_toolbar(self):
        toolbar = QToolBar("Ana Araç Çubuğu")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())
        self.addToolBar(toolbar)

        action_load = QAction("📂 Görüntü Yükle", self)
        action_load.setToolTip("Disktan PNG/JPEG/BMP seçer; sol panel orijinal, sağ panel düzenlenebilir kopya.")
        action_load.triggered.connect(self.load_image)
        toolbar.addAction(action_load)

        action_save = QAction("💾 Kaydet", self)
        action_save.setToolTip("Sağ paneldeki işlenmiş görüntüyü dosya olarak kaydeder.")
        action_save.triggered.connect(self.save_image)
        toolbar.addAction(action_save)

        action_reset = QAction("↺ Sıfırla (Reset)", self)
        action_reset.setToolTip("İşlenmiş görüntüyü yüklenen orijinale döndürür ve geçmişi temizler.")
        action_reset.triggered.connect(self.reset_image)
        toolbar.addAction(action_reset)

        action_batch = QAction("⚙ Toplu İşlem (Batch)", self)
        action_batch.setToolTip("Birden çok dosyaya aynı filtre zincirini uygular ve klasöre yazar.")
        action_batch.triggered.connect(self.batch_process)
        toolbar.addAction(action_batch)

    def setup_tabs(self):
        # 1. Ön İşleme
        tab1_widget = QWidget()
        tab1_widget.setObjectName("tabPage")
        tab1_widget.setAttribute(Qt.WA_StyledBackground, True)
        t1_layout = QVBoxLayout(tab1_widget)
        t1_layout.setSpacing(4)
        t1_layout.setContentsMargins(2, 2, 2, 2)

        g1 = QGroupBox("Temel dönüşüm")
        g1.setToolTip("Renk temsilini değiştiren temel adımlar.")
        l1 = QVBoxLayout()
        btn_rgb_gray = QPushButton("Gri Dönüşüm (RGB to Gray)")
        btn_rgb_gray.setToolTip("BGR görüntüyü gri tonlamaya çevirir.")
        btn_rgb_gray.clicked.connect(lambda: self.apply_operation("Gri Dönüşüm", preprocessing.rgb_to_gray))
        l1.addWidget(btn_rgb_gray)
        g1.setLayout(l1)

        g2 = QGroupBox("Eşikleme (Binary)")
        g2.setToolTip("Gri görüntüde seçilen eşikle siyah-beyaz görüntü üretir.")
        l2 = QVBoxLayout()
        h_bin_layout = QHBoxLayout()
        self.slider_binary = QSlider(Qt.Horizontal)
        self.slider_binary.setRange(0, 255)
        self.slider_binary.setValue(128)
        self.label_binary_val = QLabel("128")
        self.slider_binary.setToolTip("Eşik değeri: altı siyah, üstü beyaz (tipik 0–255).")
        self.slider_binary.valueChanged.connect(lambda v: self.label_binary_val.setText(str(v)))
        btn_binary = QPushButton("Binary Dönüşüm")
        btn_binary.setToolTip("Gri veya renkli girişi eşik eşitliğine göre ikili görüntüye dönüştürür.")
        btn_binary.clicked.connect(
            lambda: self.apply_operation(
                f"Binary Dönüşüm (T:{self.slider_binary.value()})",
                preprocessing.gray_to_binary,
                self.slider_binary.value(),
            )
        )
        h_bin_layout.addWidget(QLabel("Eşik (Threshold):"))
        h_bin_layout.addWidget(self.slider_binary)
        h_bin_layout.addWidget(self.label_binary_val)
        h_bin_layout.addWidget(btn_binary)
        l2.addLayout(h_bin_layout)
        g2.setLayout(l2)

        g3 = QGroupBox("Renk uzayı")
        g3.setToolTip("HSV (renk, doygunluk, parlaklık) temsiline geçiş.")
        l3 = QVBoxLayout()
        btn_rgb_hsv = QPushButton("HSV Dönüşüm")
        btn_rgb_hsv.setToolTip("RGB/BGR görüntüyü HSV renk uzayına dönüştürür.")
        btn_rgb_hsv.clicked.connect(lambda: self.apply_operation("HSV Dönüşüm", preprocessing.rgb_to_hsv))
        l3.addWidget(btn_rgb_hsv)
        g3.setLayout(l3)

        t1_one_row = QHBoxLayout()
        t1_one_row.addWidget(g1, 1)
        t1_one_row.addWidget(g2, 1)
        t1_one_row.addWidget(g3, 1)
        t1_layout.addLayout(t1_one_row)

        self.tabs.addTab(tab1_widget, "Ön İşleme")

        # 2. Geometri
        tab2_widget = QWidget()
        tab2_widget.setObjectName("tabPage")
        tab2_widget.setAttribute(Qt.WA_StyledBackground, True)
        t2_layout = QVBoxLayout(tab2_widget)
        t2_layout.setSpacing(4)
        t2_layout.setContentsMargins(2, 2, 2, 2)

        g_rot = QGroupBox("Döndürme")
        g_rot.setToolTip("Görüntüyü merkeze göre döndürür.")
        l_rot = QHBoxLayout()
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.spin_angle.setValue(45)
        self.spin_angle.setToolTip("Saat yönü pozitif derece.")
        btn_rotate = QPushButton("Döndür (Rotate)")
        btn_rotate.setToolTip("Belirtilen açıyla döndürme uygular.")
        btn_rotate.clicked.connect(
            lambda: self.apply_operation(
                f"Döndürme ({self.spin_angle.value()}°)",
                geometry.rotate_image,
                self.spin_angle.value(),
            )
        )
        l_rot.addWidget(QLabel("Döndürme Açısı (Derece):"))
        l_rot.addWidget(self.spin_angle)
        l_rot.addWidget(btn_rotate)
        g_rot.setLayout(l_rot)

        g_crop = QGroupBox("Kırpma (Crop)")
        g_crop.setToolTip("Dikdörtgen bölge keserek yeni boyut üretir.")
        l_crop = QHBoxLayout()
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 9999)
        self.spin_x.setPrefix("X: ")
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 9999)
        self.spin_y.setPrefix("Y: ")
        self.spin_w = QSpinBox()
        self.spin_w.setRange(1, 9999)
        self.spin_w.setValue(200)
        self.spin_w.setPrefix("W: ")
        self.spin_h = QSpinBox()
        self.spin_h.setRange(1, 9999)
        self.spin_h.setValue(200)
        self.spin_h.setPrefix("H: ")
        btn_crop = QPushButton("Kırp (Crop)")
        btn_crop.setToolTip("Sol üst (X,Y) ve genişlik/yükseklik ile kırpar.")
        btn_crop.clicked.connect(
            lambda: self.apply_operation(
                f"Kırpma ({self.spin_x.value()},{self.spin_y.value()},{self.spin_w.value()},{self.spin_h.value()})",
                geometry.crop_image,
                self.spin_x.value(),
                self.spin_y.value(),
                self.spin_w.value(),
                self.spin_h.value(),
            )
        )
        for w in [self.spin_x, self.spin_y, self.spin_w, self.spin_h, btn_crop]:
            l_crop.addWidget(w)
        g_crop.setLayout(l_crop)
        t2_row_rc = QHBoxLayout()
        t2_row_rc.addWidget(g_rot, 1)
        t2_row_rc.addWidget(g_crop, 1)
        t2_layout.addLayout(t2_row_rc)

        g_zoom = QGroupBox("Ölçekleme (Zoom)")
        g_zoom.setToolTip("Görüntüyü yeniden boyutlandırır (oran korunur).")
        l_zoom = QHBoxLayout()
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(1, 40)
        self.slider_zoom.setValue(10)
        self.slider_zoom.setToolTip("0.1× ile 4.0× arası ölçek (değer÷10).")
        self.label_zoom = QLabel("1.0x")
        self.slider_zoom.valueChanged.connect(lambda v: self.label_zoom.setText(f"{v/10.0}x"))
        btn_zoom = QPushButton("Yeniden Ölçeklendir (Zoom)")
        btn_zoom.setToolTip("İnterpolasyonlu yeniden ölçeklendirme.")
        btn_zoom.clicked.connect(
            lambda: self.apply_operation(
                f"Zoom ({self.slider_zoom.value()/10.0}x)",
                geometry.zoom_image,
                self.slider_zoom.value() / 10.0,
            )
        )
        l_zoom.addWidget(QLabel("Boyut Çarpanı (Ölçek):"))
        l_zoom.addWidget(self.slider_zoom)
        l_zoom.addWidget(self.label_zoom)
        l_zoom.addWidget(btn_zoom)
        g_zoom.setLayout(l_zoom)
        t2_layout.addWidget(g_zoom)

        self.tabs.addTab(tab2_widget, "Geometrik Düzeltme")

        # 3. İyileştirme
        tab3_widget = QWidget()
        tab3_widget.setObjectName("tabPage")
        tab3_widget.setAttribute(Qt.WA_StyledBackground, True)
        t3_layout = QVBoxLayout(tab3_widget)
        t3_layout.setSpacing(4)
        t3_layout.setContentsMargins(2, 2, 2, 2)

        g_hist = QGroupBox("Histogram ve analiz")
        g_hist.setToolTip("Parlaklık dağılımını analiz eder ve germe uygular.")
        lh = QVBoxLayout()
        btn_hist_show = QPushButton("Histogram Analizini Göster (Matplotlib)")
        btn_hist_show.setToolTip("Mevcut görüntünün histogramını yeni pencerede çizer.")
        btn_hist_show.clicked.connect(
            lambda: enhancement.plot_histogram(self.current_image)
            if self.current_image is not None
            else QMessageBox.warning(self, "Hata", "Lütfen önce görüntü yükleyin!")
        )
        btn_hist_stretch = QPushButton("Histogram Germe (Stretching)")
        btn_hist_stretch.setToolTip("Minimum–maksimum parlaklığı 0–255 aralığına yayar.")
        btn_hist_stretch.clicked.connect(lambda: self.apply_operation("Histogram Germe", enhancement.histogram_stretching))
        lh.addWidget(btn_hist_show)
        lh.addWidget(btn_hist_stretch)
        g_hist.setLayout(lh)

        g_ton = QGroupBox("Kontrast ve gamma")
        g_ton.setToolTip("Lineer kontrast ve güç-law gamma düzeltmesi.")
        lt = QVBoxLayout()
        h_contrast = QHBoxLayout()
        self.slider_cont = QSlider(Qt.Horizontal)
        self.slider_cont.setRange(1, 30)
        self.slider_cont.setValue(10)
        self.slider_cont.setToolTip("Kontrast çarpanı (1.0 = nötr).")
        self.label_cont = QLabel("1.0x")
        self.slider_cont.valueChanged.connect(lambda v: self.label_cont.setText(f"{v/10.0}x"))
        btn_contrast = QPushButton("Kontrast Artır")
        btn_contrast.setToolTip("Merkez 128 etrafında kontrast artırır.")
        btn_contrast.clicked.connect(
            lambda: self.apply_operation(
                f"Kontrast Artırma ({self.slider_cont.value()/10.0}x)",
                enhancement.contrast_enhancement,
                self.slider_cont.value() / 10.0,
            )
        )
        h_contrast.addWidget(QLabel("Kontrast Kat Sayısı:"))
        h_contrast.addWidget(self.slider_cont)
        h_contrast.addWidget(self.label_cont)
        h_contrast.addWidget(btn_contrast)
        lt.addLayout(h_contrast)

        h_gamma = QHBoxLayout()
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(1, 50)
        self.slider_gamma.setValue(10)
        self.slider_gamma.setToolTip("γ<1 aydınlatır, γ>1 koyulaştırır (1.0 nötr).")
        self.label_gamma = QLabel("γ = 1.0")
        self.slider_gamma.valueChanged.connect(lambda v: self.label_gamma.setText(f"γ = {v / 10.0:.1f}"))
        btn_gamma = QPushButton("Gamma Düzeltme Uygula")
        btn_gamma.setToolTip("Görüntüyü γ ile yeniden eşler (0.1–5.0).")
        btn_gamma.clicked.connect(
            lambda: self.apply_operation(
                f"Gamma Correction (γ={self.slider_gamma.value() / 10.0:.1f})",
                enhancement.gamma_correction,
                self.slider_gamma.value() / 10.0,
            )
        )
        h_gamma.addWidget(QLabel("Gamma Correction (γ):"))
        h_gamma.addWidget(self.slider_gamma)
        h_gamma.addWidget(self.label_gamma)
        h_gamma.addWidget(btn_gamma)
        lt.addLayout(h_gamma)
        g_ton.setLayout(lt)
        t3_row_a = QHBoxLayout()
        t3_row_a.addWidget(g_hist, 1)
        t3_row_a.addWidget(g_ton, 1)
        t3_layout.addLayout(t3_row_a)

        g_noise = QGroupBox("Gürültü")
        g_noise.setToolTip("Yapay tuz-biber gürültüsü ekler (test ve filtre denemeleri).")
        ln = QHBoxLayout()
        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setRange(1, 20)
        self.slider_noise.setValue(5)
        self.slider_noise.setToolTip("Bozulmuş piksel yüzdesi (1–20).")
        self.label_noise = QLabel("%5")
        self.slider_noise.valueChanged.connect(lambda v: self.label_noise.setText(f"%{v}"))
        btn_noise = QPushButton("Tuz / Biber Gürültüsü Ekle (Salt & Pepper)")
        btn_noise.setToolTip("Rastgele siyah ve beyaz noktalar ekler.")
        btn_noise.clicked.connect(
            lambda: self.apply_operation(
                f"Gürültü Ekle (%{self.slider_noise.value()})",
                enhancement.add_salt_pepper_noise,
                self.slider_noise.value(),
            )
        )
        ln.addWidget(QLabel("Bozulma Miktarı:"))
        ln.addWidget(self.slider_noise)
        ln.addWidget(self.label_noise)
        ln.addWidget(btn_noise)
        g_noise.setLayout(ln)

        g_two = QGroupBox("İki görüntü ile aritmetik")
        g_two.setToolTip("İkinci dosyayı seçerek ortalama, toplama veya bölme.")
        l2i = QVBoxLayout()
        btn_avg = QPushButton("Image Averaging (Ortalama / Gürültü Azaltma) - 2. Görüntüyü Seç")
        btn_avg.setToolTip("İkinci görüntüyle piksel ortalaması alır (boyut uyumu gerekir).")
        btn_avg.clicked.connect(self.do_image_averaging)
        btn_add = QPushButton("Aritmetik İşlem: İki Görüntüyü Ekle (Harmanla) - 2. Görüntüyü Seç")
        btn_add.setToolTip("İki görüntüyü toplayarak harmanlar.")
        btn_add.clicked.connect(self.do_image_addition)
        btn_div = QPushButton("Aritmetik İşlem: İki Görüntüyü Böl (Fark/Normalize) - 2. Görüntüyü Seç")
        btn_div.setToolTip("İkinci görüntüye böler (normalize karşılaştırma).")
        btn_div.clicked.connect(self.do_image_division)
        l2i.addWidget(btn_avg)
        l2i.addWidget(btn_add)
        l2i.addWidget(btn_div)
        g_two.setLayout(l2i)
        t3_row_b = QHBoxLayout()
        t3_row_b.addWidget(g_noise, 1)
        t3_row_b.addWidget(g_two, 1)
        t3_layout.addLayout(t3_row_b)

        self.tabs.addTab(tab3_widget, "Görüntü İyileştirme")

        # 4. Filtreleme
        tab4_widget = QWidget()
        tab4_widget.setObjectName("tabPage")
        tab4_widget.setAttribute(Qt.WA_StyledBackground, True)
        t4_layout = QVBoxLayout(tab4_widget)
        t4_layout.setSpacing(4)
        t4_layout.setContentsMargins(2, 2, 2, 2)

        g_smooth = QGroupBox("Yumuşatma (düşük geçiren)")
        g_smooth.setToolTip("Yerel ortalama ve median ile gürültü azaltma.")
        ls = QVBoxLayout()
        h_mean = QHBoxLayout()
        self.combo_mean = QComboBox()
        self.combo_mean.addItems(["3", "5", "7", "9"])
        self.combo_mean.setToolTip("Kare çekirdek boyutu (tek sayı).")
        btn_mean = QPushButton("Mean Filtre Uygula")
        btn_mean.setToolTip("Komşu piksellerin ortalaması (kutu bulanıklaştırma).")
        btn_mean.clicked.connect(
            lambda: self.apply_operation(
                f"Mean Filtre ({self.combo_mean.currentText()}x{self.combo_mean.currentText()})",
                filtering.mean_filter,
                int(self.combo_mean.currentText()),
            )
        )
        h_mean.addWidget(QLabel("Kernel Boyutu (KxK):"))
        h_mean.addWidget(self.combo_mean)
        h_mean.addWidget(btn_mean)
        ls.addLayout(h_mean)

        h_median = QHBoxLayout()
        self.combo_median = QComboBox()
        self.combo_median.addItems(["3", "5", "7", "9"])
        self.combo_median.setToolTip("Median pencere boyutu.")
        btn_median = QPushButton("Median Filtre Uygula")
        btn_median.setToolTip("Tuz-biber gürültüsüne dayanıklı yumuşatma.")
        btn_median.clicked.connect(
            lambda: self.apply_operation(
                f"Median Filtre ({self.combo_median.currentText()}x{self.combo_median.currentText()})",
                filtering.median_filter,
                int(self.combo_median.currentText()),
            )
        )
        h_median.addWidget(QLabel("Kernel Boyutu (KxK):"))
        h_median.addWidget(self.combo_median)
        h_median.addWidget(btn_median)
        ls.addLayout(h_median)
        g_smooth.setLayout(ls)

        g_sharp = QGroupBox("Keskinleştirme")
        g_sharp.setToolTip("Unsharp mask ile kenar vurgusu.")
        lsh = QHBoxLayout()
        self.slider_unsharp = QSlider(Qt.Horizontal)
        self.slider_unsharp.setRange(5, 30)
        self.slider_unsharp.setValue(10)
        self.slider_unsharp.setToolTip("Keskinleştirme gücü (0.5–3.0).")
        self.label_unsharp = QLabel("1.0")
        self.slider_unsharp.valueChanged.connect(lambda v: self.label_unsharp.setText(str(v / 10.0)))
        btn_unsharp = QPushButton("Unsharp Masking (Keskinleştirme)")
        btn_unsharp.setToolTip("Bulanık çıkarma ile detay güçlendirme.")
        btn_unsharp.clicked.connect(
            lambda: self.apply_operation(
                f"Unsharp Masking ({self.slider_unsharp.value()/10.0})",
                filtering.unsharp_mask,
                self.slider_unsharp.value() / 10.0,
            )
        )
        lsh.addWidget(QLabel("Güç (Strength):"))
        lsh.addWidget(self.slider_unsharp)
        lsh.addWidget(self.label_unsharp)
        lsh.addWidget(btn_unsharp)
        g_sharp.setLayout(lsh)
        t4_row = QHBoxLayout()
        t4_row.addWidget(g_smooth, 1)
        t4_row.addWidget(g_sharp, 1)
        t4_layout.addLayout(t4_row)

        self.tabs.addTab(tab4_widget, "Gürültü & Keskinlik")

        # 5. Analiz
        tab5_widget = QWidget()
        tab5_widget.setObjectName("tabPage")
        tab5_widget.setAttribute(Qt.WA_StyledBackground, True)
        t5_layout = QVBoxLayout(tab5_widget)
        t5_layout.setSpacing(4)
        t5_layout.setContentsMargins(2, 2, 2, 2)

        g_edge = QGroupBox("Kenar çıkarma")
        g_edge.setToolTip("Gradyan tabanlı kenar haritası.")
        le = QVBoxLayout()
        btn_prewitt = QPushButton("Prewitt Kenar Bulma (Edge Detection)")
        btn_prewitt.setToolTip("Prewitt operatörleri ile kenar yönü ve büyüklüğü.")
        btn_prewitt.clicked.connect(lambda: self.apply_operation("Prewitt Kenar Çıkarma", analysis.prewitt_edge_detection))
        le.addWidget(btn_prewitt)
        g_edge.setLayout(le)

        g_morph = QGroupBox("Morfolojik işlemler")
        g_morph.setToolTip("Şekil çekirdeği ile erozyon, genişletme, açma ve kapama.")
        lm = QVBoxLayout()
        h_kernel = QHBoxLayout()
        h_kernel.addWidget(QLabel("Morfolojik Hedef Kernel Boyutu:"))
        self.combo_morph = QComboBox()
        self.combo_morph.addItems(["3", "5"])
        self.combo_morph.setToolTip("Yapısal eleman pencere boyutu.")
        h_kernel.addWidget(self.combo_morph)
        h_kernel.addWidget(QLabel("Kernel Şekli:"))
        self.combo_morph_shape = QComboBox()
        self.combo_morph_shape.addItems(["rect", "cross", "ellipse"])
        self.combo_morph_shape.setToolTip("rect: dolu kare; cross: haç; ellipse: dolu daire maskesi.")
        h_kernel.addWidget(self.combo_morph_shape)
        h_kernel.addStretch()
        lm.addLayout(h_kernel)

        btn_erosion = QPushButton("Erosion (Aşındırma)")
        btn_erosion.setToolTip("Ön plandaki beyaz bölgeleri inceltir, gürültüyü soyar.")
        btn_erosion.clicked.connect(
            lambda: self.apply_operation(
                f"Erosion ({self.combo_morph_shape.currentText()}, {self.combo_morph.currentText()}x{self.combo_morph.currentText()})",
                analysis.erosion,
                self._morph_structuring_element(),
            )
        )
        lm.addWidget(btn_erosion)

        btn_dilation = QPushButton("Dilation (Genişletme)")
        btn_dilation.setToolTip("Beyaz bölgeleri kalınlaştırır, delikleri küçültür.")
        btn_dilation.clicked.connect(
            lambda: self.apply_operation(
                f"Dilation ({self.combo_morph_shape.currentText()}, {self.combo_morph.currentText()}x{self.combo_morph.currentText()})",
                analysis.dilation,
                self._morph_structuring_element(),
            )
        )
        lm.addWidget(btn_dilation)

        btn_opening = QPushButton("Opening (Açma İşlemi)")
        btn_opening.setToolTip("Önce erozyon sonra genişletme; küçük gürültü lekelerini temizler.")
        btn_opening.clicked.connect(
            lambda: self.apply_operation(
                f"Opening ({self.combo_morph_shape.currentText()}, {self.combo_morph.currentText()}x{self.combo_morph.currentText()})",
                analysis.opening,
                self._morph_structuring_element(),
            )
        )
        lm.addWidget(btn_opening)

        btn_closing = QPushButton("Closing (Kapatma İşlemi)")
        btn_closing.setToolTip("Önce genişletme sonra erozyon; küçük delikleri kapatır.")
        btn_closing.clicked.connect(
            lambda: self.apply_operation(
                f"Closing ({self.combo_morph_shape.currentText()}, {self.combo_morph.currentText()}x{self.combo_morph.currentText()})",
                analysis.closing,
                self._morph_structuring_element(),
            )
        )
        lm.addWidget(btn_closing)
        g_morph.setLayout(lm)
        t5_split = QHBoxLayout()
        t5_split.addWidget(g_edge, 1)
        t5_split.addWidget(g_morph, 2)
        t5_layout.addLayout(t5_split)

        self.tabs.addTab(tab5_widget, "Analiz & Morfoloji")

    # ==========================
    # UYGULAMA MANTIK (LOGIC) FONKSİYONLARI 
    # ==========================

    def clear_history(self):
        self.history.clear()
        self.history_list.clear()
        self.update_panel_metadata()

    def update_panel_metadata(self):
        def ch_text(img):
            if img is None:
                return "—"
            if len(img.shape) == 3:
                return f"{img.shape[2]} kanal (BGR)"
            return "Tek kanal (gri/binary)"

        def meta_card(title, lines):
            esc_lines = "<br/>".join(html.escape(L) for L in lines)
            return (
                "<div style='line-height:1.42'>"
                "<span style='font-weight:600;color:#e8eaf4;font-size:12px;letter-spacing:0.02em'>"
                f"{html.escape(title)}"
                "</span>"
                "<span style='display:block;margin-top:6px;color:#9ca3b8;font-size:11px'>"
                f"{esc_lines}"
                "</span>"
                "</div>"
            )

        if self.original_image is None:
            self.lbl_orig_meta.setText(
                meta_card(
                    "Kaynak görüntü",
                    ["Dosya: —", "Çözünürlük: —", "Kanallar: —", "Son işlem: —"],
                )
            )
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "—"
            ho, wo = self.original_image.shape[:2]
            self.lbl_orig_meta.setText(
                meta_card(
                    "Kaynak görüntü",
                    [
                        f"Dosya: {fn}",
                        f"Çözünürlük: {wo} × {ho}",
                        f"Kanallar: {ch_text(self.original_image)}",
                        "Son işlem: —",
                    ],
                )
            )

        if self.current_image is None:
            self.lbl_proc_meta.setText(
                meta_card(
                    "İşlenmiş önizleme",
                    ["Kaynak dosya: —", "Çözünürlük: —", "Kanallar: —", "Son işlem: —"],
                )
            )
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "—"
            hc, wc = self.current_image.shape[:2]
            last = self.history[-1][0] if self.history else "Henüz işlem yok"
            self.lbl_proc_meta.setText(
                meta_card(
                    "İşlenmiş önizleme",
                    [
                        f"Kaynak dosya: {fn}",
                        f"Çözünürlük: {wc} × {hc}",
                        f"Kanallar: {ch_text(self.current_image)}",
                        f"Son işlem: {last}",
                    ],
                )
            )

    def _morph_structuring_element(self):
        return analysis.get_structuring_element(
            self.combo_morph_shape.currentText(),
            int(self.combo_morph.currentText()),
        )

    def apply_operation(self, op_name, module_func, *args):
        if self.current_image is None:
            QMessageBox.warning(self, "Hata", "Lütfen önce bir görüntü yükleyin!")
            return
            
        try:
            # Geri alma (Undo) mekanizması için mevcut görüntüyü (kopya) bellekte sakla
            prev_state = self.current_image.copy()
            
            # Seçilen operasyonu mevcut görüntü üzerinde argümanlarıyla çağır
            result = module_func(self.current_image, *args)
            
            # Eger fonksiyon geriye degistirilen bir NumPy matrisi döndürdüyse listeye ekle
            if result is not None:
                self.history.append((op_name, module_func, args, prev_state))
                self.history_list.addItem(f"{len(self.history)}. {op_name}")
                self.current_image = result
                self.update_image_display(self.current_image, self.right_label)
                self.update_panel_metadata()

        except NotImplementedError as e:
            QMessageBox.information(self, "1. Aşama Bilgisi", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"İşlem sırasında beklenmeyen bir matris (boyut vb) hatası oluştu:\n{str(e)}")

    def undo_operation_from_list(self, item):
        # Listede çift tıklanan aşamaya geri döner
        row = self.history_list.row(item)
        self.restore_history_state(row)

    def undo_operation(self):
        # Son islemi geri alır
        if not self.history:
            return
        last_index = len(self.history) - 1
        self.restore_history_state(last_index)

    def restore_history_state(self, row_index):
        if row_index < 0 or row_index >= len(self.history):
            return
            
        # İstenen aşamanın kaydedilmiş orijinal matrisine dön
        prev_state = self.history[row_index][3]
        
        # Seçilen işlem dahil olmak üzere o andan sonraki "gelecek" geçmişi temizle
        self.history = self.history[:row_index]
        while self.history_list.count() > row_index:
            self.history_list.takeItem(row_index)
            
        self.current_image = prev_state.copy()
        self.update_image_display(self.current_image, self.right_label)
        self.update_panel_metadata()

    def do_image_averaging(self):
        if self.current_image is None: 
            QMessageBox.warning(self, "Hata", "Lütfen önce ana (1.) görüntüyü yükleyin!")
            return
        
        file_name, _ = QFileDialog.getOpenFileName(self, "2. Görüntüyü Seç (Averaging)", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # OpenCV sadece okuma amaciyla serbest! (Türkçe yol uyumlu)
            img2 = safe_imread(file_name)
            if img2 is not None:
                self.apply_operation("Image Averaging (Ortalama Alma)", enhancement.image_averaging, img2)
            else:
                QMessageBox.warning(self, "Hata", "İkinci görüntü okunamadı!")

    def do_image_addition(self):
        if self.current_image is None: 
            QMessageBox.warning(self, "Hata", "Lütfen önce ana (1.) görüntüyü yükleyin!")
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "2. Görüntüyü Seç (Ekleme)", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            img2 = safe_imread(file_name)
            if img2 is not None:
                self.apply_operation("Görüntü Ekleme (Aritmetik)", geometry.add_images, img2)
            else:
                QMessageBox.warning(self, "Hata", "İkinci görüntü okunamadı!")

    def do_image_division(self):
        if self.current_image is None: 
            QMessageBox.warning(self, "Hata", "Lütfen önce ana (1.) görüntüyü yükleyin!")
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "2. Görüntüyü Seç (Bölme)", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            img2 = safe_imread(file_name)
            if img2 is not None:
                self.apply_operation("Görüntü Bölme (Aritmetik)", geometry.divide_images, img2)
            else:
                QMessageBox.warning(self, "Hata", "İkinci görüntü okunamadı!")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            # Opencv imread SERBEST Işığı altında okunuyor (Türkçe karakter destekli)
            self.original_image = safe_imread(file_name)
            if self.original_image is not None:
                self.image_path = file_name
                self.current_image = self.original_image.copy()
                self.history.clear()
                self.history_list.clear()

                self.update_image_display(self.original_image, self.left_label)
                self.update_image_display(self.current_image, self.right_label)
                self.update_panel_metadata()
            else:
                self.image_path = None
                QMessageBox.warning(self, "Hata", "Görüntü okunamadı veya dosya bozuk!")

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Hata", "Kaydedilecek herhangi bir görüntü yok!")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "İşlenmiş Görüntüyü Kaydet", "islenmis_goruntu.jpg", "JPEG Images (*.jpg);;PNG Images (*.png);;BMP Images (*.bmp)")
        if file_name:
            # Opencv imwrite kısıtlamalara uyarak veri yazmak için kullanılıyor (Türkçe karakter destekli)
            safe_imwrite(file_name, self.current_image)
            QMessageBox.information(self, "Başarılı", f"Dosya başarıyla kaydedildi:\n{file_name}")

    def reset_image(self):
        # İşlemleri sıfırlayıp başa alma
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.history.clear()
            self.history_list.clear()
            self.update_image_display(self.current_image, self.right_label)
            self.update_panel_metadata()

    def batch_process(self):
        """Kullanıcının seçtiği işlemleri içeren yeni Toplu İşlem penceresini açar"""
        dialog = BatchProcessDialog(self)
        if dialog.exec_():
            if not dialog.input_files or not dialog.output_folder:
                QMessageBox.warning(self, "Uyarı", "Lütfen hem işlenecek görüntüleri hem de kaydedilecek çıktı klasörünü seçin!")
                return
            
            ops = dialog.get_operations()
            if not ops:
                QMessageBox.warning(self, "Uyarı", "Hiçbir işlem/filtre seçmediniz!")
                return
                
            try:
                processed_count = 0
                last_processed_img = None
                
                for file_path in dialog.input_files:
                    filename = os.path.basename(file_path)
                    original_img = safe_imread(file_path)
                    if original_img is None: continue
                    
                    # Ana ekranda canlı (orijinal) göster
                    self.update_image_display(original_img, self.left_label)
                    QApplication.processEvents()

                    img = original_img.copy()
                    
                    # Seçilen işlemleri sırasıyla bu görüntüye uygula
                    for op_func, op_args in ops:
                        res = op_func(img, *op_args)
                        if res is not None:
                            img = res
                    
                    # Çıktı klasörüne kaydet
                    out_path = os.path.join(dialog.output_folder, filename)
                    safe_imwrite(out_path, img)
                    processed_count += 1
                    
                    last_processed_img = img
                    # İşlenmiş halini ana ekranda canlı (işlenmiş) göster
                    self.update_image_display(img, self.right_label)
                    QApplication.processEvents()
                        
                QMessageBox.information(self, "Başarılı", f"Toplu batch işlemi tamamlandı!\nBaşarıyla işlenen dosya sayısı: {processed_count}")
                
                # Kullanıcıya işlemin sonucunu (son dosya üzerinden) pop-up olarak göster
                if processed_count > 0 and last_processed_img is not None:
                    preview_dialog = QDialog(self)
                    preview_dialog.setWindowTitle("Örnek İşlem Sonucu (Son Fotoğraf)")
                    preview_dialog.resize(700, 700)
                    pv_layout = QVBoxLayout(preview_dialog)
                    
                    lbl_info = QLabel("Seçtiğiniz filtrelerin klasördeki fotoğraflara uygulanmış son hali (Örnek Önizleme):")
                    lbl_info.setStyleSheet("font-weight: bold; font-size: 14px;")
                    pv_layout.addWidget(lbl_info)
                    
                    lbl_preview = QLabel()
                    lbl_preview.setAlignment(Qt.AlignCenter)
                    lbl_preview.setStyleSheet("border: 2px solid #555555; background-color: #1e1e2e;")
                    self.update_image_display(last_processed_img, lbl_preview)
                    pv_layout.addWidget(lbl_preview, stretch=1)
                    
                    btn_ok = QPushButton("Kapat")
                    btn_ok.clicked.connect(preview_dialog.accept)
                    btn_ok.setStyleSheet("background-color: #4a90e2; padding: 10px; font-weight: bold;")
                    pv_layout.addWidget(btn_ok)
                    
                    preview_dialog.exec_()
                    
            except Exception as e:
                QMessageBox.critical(self, "Toplu İşlem Hatası", f"Toplu işlem başarısız oldu:\n{str(e)}")

    def update_image_display(self, img_array, label):
        """
        Matris Array'i QImage'e ve ardından QPixmap'e dönüştürerek ekranda orantılı biçimde gösterir.
        CV2'nin yasaklı cvtColor fonksiyonuna alternatif olarak NumPy'ın dilimlemesi kullanılmıştır.
        """
        if img_array is None: return
        
        # Olası float matris hatalarına karşı güvene alma
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        h, w = img_array.shape[:2]
        
        if len(img_array.shape) == 3:
            # OpenCV resmi BGR okur. Görüntülemek için RGB yapmalıyız.
            # cvtColor kullanmak yasak olduğu için array slicing yapılarak ters çevriliyor [::-1]
            rgb_img = img_array[:, :, ::-1].copy() 
            bytes_per_line = 3 * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Gri ya da Binary Image
            bytes_per_line = w
            gr_img = np.ascontiguousarray(img_array)
            q_img = QImage(gr_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        # QLabel geometrisi henüz hesaplanmadıysa width/height 0 dönebilir. Qt'de scaled(..., 0) veya (0, ...)
        # tek boyutu kullanıp diğerini en-boy oranıyla çözer; sonuç tam çözünürlükte devasa bir pixmap olabilir,
        # QLabel layout'ta şişer ve alt sekme bandını sıkıştırır. Bu yüzden her iki boyutu da en az 1'e sabitle.
        lw = max(1, label.width())
        lh = max(1, label.height())
        pixmap_scaled = pixmap.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap_scaled)

    def resizeEvent(self, event):
        # Ekran boyutu değiştiğinde fotoğrafları güncel Label'a göre yeniden ölçekle
        if self.original_image is not None:
            self.update_image_display(self.original_image, self.left_label)
        if self.current_image is not None:
            self.update_image_display(self.current_image, self.right_label)
        super().resizeEvent(event)

MODERN_STYLE = """
/* —— Palet: kurumsal koyu (slate / mavi-gri) —— */
QMainWindow, QDialog {
    background-color: #14151c;
}
QWidget {
    color: #ececf3;
    font-family: 'Segoe UI', 'Inter', 'SF Pro Text', system-ui, sans-serif;
    font-size: 13px;
}
QLabel { color: #dce0eb; background-color: transparent; }

QLabel#panelMeta {
    padding: 10px 12px;
    background-color: #1e2030;
    border: 1px solid #2e3348;
    border-radius: 12px;
}

QLabel#panelImage {
    border: 1px dashed #3d4a63;
    border-radius: 14px;
    background-color: #1a1c28;
    padding: 12px;
}

QFrame#tabDivider {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2a3348, stop:0.5 #5d8cff, stop:1 #2a3348);
    border: none;
    border-radius: 1px;
    max-height: 2px;
}

QGroupBox#historyCard {
    border: 1px solid #2e3348;
    border-radius: 14px;
    margin-top: 10px;
    padding: 12px 12px 10px 12px;
    font-weight: 600;
    font-size: 12px;
    color: #eef0f7;
    background-color: #1e2030;
}
QGroupBox#historyCard::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 4px 12px;
    color: #ffffff;
    background-color: #2d354d;
    border-radius: 6px;
    font-size: 11px;
    letter-spacing: 0.03em;
}

QWidget#tabPage {
    background-color: #181a26;
    color: #c8cdd9;
}
QWidget#tabPage QLabel { color: #c8cdd9; background-color: transparent; }

QGroupBox {
    border: 1px solid #323848;
    border-radius: 10px;
    margin-top: 6px;
    padding: 8px 10px 9px 10px;
    font-weight: 600;
    font-size: 12px;
    color: #eff1f7;
    background-color: #222534;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 3px 10px;
    left: 10px;
    color: #ffffff;
    background-color: #343b52;
    border-radius: 5px;
    font-size: 11px;
}
QTabWidget QGroupBox {
    margin-top: 4px;
    padding: 6px 8px 7px 8px;
}
QTabWidget QPushButton {
    padding: 5px 11px;
    min-height: 14px;
    font-size: 12px;
}

QPushButton {
    background-color: #2a3145;
    border: 1px solid #3d4660;
    border-radius: 10px;
    padding: 8px 14px;
    color: #f4f6fb;
    min-height: 16px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #343d56;
    border: 1px solid #5d8cff;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #232938;
    border: 1px solid #4a7ae8;
}

QTabWidget::pane {
    border: 1px solid #2e3348;
    border-radius: 12px;
    top: -1px;
    background-color: #181a26;
    padding: 6px;
}

QTabBar::tab {
    background-color: #1e2230;
    color: #8b92a8;
    padding: 9px 18px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 4px;
    font-size: 12px;
    min-height: 18px;
}
QTabBar::tab:selected {
    background-color: #181a26;
    color: #ffffff;
    font-weight: 600;
    border-bottom: 3px solid #5d8cff;
    margin-bottom: -1px;
}
QTabBar::tab:hover:!selected {
    background-color: #262a3a;
    color: #cdd2e0;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #12141d;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e8ebf4;
    border: 1px solid #5d8cff;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border: 1px solid #8fb4ff;
}

QListWidget {
    background-color: #151721;
    border: 1px solid #2e3348;
    border-radius: 12px;
    padding: 8px;
    outline: none;
    font-size: 12px;
}
QListWidget::item {
    padding: 10px 12px;
    border-radius: 8px;
    margin: 3px 0;
    color: #d8dce8;
}
QListWidget::item:selected {
    background-color: #2a3f6e;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background-color: #252838;
}

QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #222534;
    border: 1px solid #3d4660;
    border-radius: 8px;
    padding: 6px 10px;
    color: #ececf3;
    min-height: 22px;
    selection-background-color: #3d5a9e;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #5d8cff;
}
QComboBox QAbstractItemView {
    background-color: #1e2230;
    color: #ececf3;
    selection-background-color: #2a3f6e;
    selection-color: #ffffff;
    border: 1px solid #2e3348;
    border-radius: 8px;
    padding: 4px;
}

QToolBar {
    background-color: #1a1c28;
    border: none;
    border-bottom: 1px solid #2e3348;
    spacing: 6px;
    padding: 10px 14px;
}
QToolButton {
    background-color: transparent;
    color: #dce0eb;
    padding: 9px 14px;
    border-radius: 10px;
    margin: 0 3px;
    border: 1px solid transparent;
}
QToolButton:hover {
    background-color: #252838;
    border: 1px solid #3d4660;
    color: #ffffff;
}
QToolButton:pressed {
    background-color: #1e2230;
    border: 1px solid #5d8cff;
}

QCheckBox { spacing: 10px; font-size: 13px; color: #dce0eb; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #4a546e;
    border-radius: 5px;
    background-color: #222534;
}
QCheckBox::indicator:hover { border: 1px solid #5d8cff; }
QCheckBox::indicator:checked {
    background-color: #5d8cff;
    border: 1px solid #5d8cff;
}

QMessageBox { background-color: #14151c; }
QMessageBox QLabel { color: #ececf3; font-size: 13px; }

QScrollBar:vertical {
    background: #151721;
    width: 11px;
    border-radius: 5px;
    margin: 4px 2px 4px 2px;
}
QScrollBar::handle:vertical {
    background: #3d455c;
    border-radius: 5px;
    min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: #4d5670; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #151721;
    height: 11px;
    border-radius: 5px;
    margin: 2px 4px 2px 4px;
}
QScrollBar::handle:horizontal {
    background: #3d455c;
    border-radius: 5px;
    min-width: 32px;
}
QScrollBar::handle:horizontal:hover { background: #4d5670; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Koyu Temaya (Dark Mode) yakın modern OS stili Fusion
    app.setStyle("Fusion")
    app.setStyleSheet(MODERN_STYLE)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())