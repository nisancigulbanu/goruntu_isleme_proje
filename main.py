import sys
import os
import html
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTabWidget, QSlider, QSpinBox,
    QMessageBox, QFileDialog, QToolBar, QAction, QListWidget,
    QComboBox, QGroupBox, QSizePolicy, QDialog, QCheckBox, QDoubleSpinBox,
    QFrame, QShortcut, QSplitter, QScrollArea, QAbstractScrollArea,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def safe_imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_imwrite(path, img):
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


class DummyModule:
    def __getattr__(self, name):
        def w(*a, **k):
            raise NotImplementedError(name)
        return w


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


def gaussian_blur_sigma(image, sigma):
    return filtering.convolve2d(image, filtering.manuel_gaussian_kernel(float(sigma)))


class BatchProcessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dijital Arşivci: Tarihi Belge ve Fotoğraf Restorasyonu")
        self.resize(500, 550)
        self.input_files = []
        self.output_folder = ""
        dlg_layout = QVBoxLayout(self)
        grp_io = QGroupBox("Girdi ve çıktı")
        vio = QVBoxLayout()
        h_in = QHBoxLayout()
        btn_in = QPushButton("İşlenecek görüntüleri seç…")
        btn_in.clicked.connect(self.select_in)
        self.lbl_in = QLabel("Henüz dosya seçilmedi")
        h_in.addWidget(btn_in)
        h_in.addWidget(self.lbl_in, 1)
        vio.addLayout(h_in)
        h_out = QHBoxLayout()
        btn_out = QPushButton("Çıktı klasörü seç…")
        btn_out.clicked.connect(self.select_out)
        self.lbl_out = QLabel("Henüz klasör seçilmedi")
        h_out.addWidget(btn_out)
        h_out.addWidget(self.lbl_out, 1)
        vio.addLayout(h_out)
        grp_io.setLayout(vio)
        dlg_layout.addWidget(grp_io)
        self.group_ops = QGroupBox("Uygulanacak filtreleri seçin (sırayla)")
        self.vbox_ops = QVBoxLayout()
        self.chk_gray = QCheckBox("Gri Dönüşüm (Grayscale)")
        self.vbox_ops.addWidget(self.chk_gray)
        h_bin = QHBoxLayout()
        self.chk_binary = QCheckBox("Binary Dönüşüm")
        self.spin_binary = QSpinBox()
        self.spin_binary.setRange(0, 255)
        self.spin_binary.setValue(128)
        h_bin.addWidget(self.chk_binary)
        h_bin.addWidget(QLabel("Eşik:"))
        h_bin.addWidget(self.spin_binary)
        h_bin.addStretch()
        self.vbox_ops.addLayout(h_bin)
        self.chk_hist = QCheckBox("Histogram Germe")
        self.vbox_ops.addWidget(self.chk_hist)
        h_cont = QHBoxLayout()
        self.chk_contrast = QCheckBox("Kontrast Artır")
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.1, 5.0)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setValue(1.5)
        h_cont.addWidget(self.chk_contrast)
        h_cont.addWidget(self.spin_contrast)
        h_cont.addStretch()
        self.vbox_ops.addLayout(h_cont)
        h_med = QHBoxLayout()
        self.chk_median = QCheckBox("Median")
        self.combo_median = QComboBox()
        self.combo_median.addItems(["3", "5", "7", "9"])
        h_med.addWidget(self.chk_median)
        h_med.addWidget(self.combo_median)
        self.vbox_ops.addLayout(h_med)
        h_uns = QHBoxLayout()
        self.chk_unsharp = QCheckBox("Unsharp")
        self.spin_unsharp = QDoubleSpinBox()
        self.spin_unsharp.setRange(0.1, 5.0)
        self.spin_unsharp.setValue(1.5)
        h_uns.addWidget(self.chk_unsharp)
        h_uns.addWidget(self.spin_unsharp)
        self.vbox_ops.addLayout(h_uns)
        self.chk_prewitt = QCheckBox("Prewitt")
        self.vbox_ops.addWidget(self.chk_prewitt)
        self.group_ops.setLayout(self.vbox_ops)
        dlg_layout.addWidget(self.group_ops)
        self.btn_start = QPushButton("Toplu İşlemi Başlat")
        self.btn_start.clicked.connect(self.accept)
        dlg_layout.addWidget(self.btn_start)

    def select_in(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Görüntüleri Seç", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.input_files = files
            self.lbl_in.setText(f"{len(files)} dosya")

    def select_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı klasörü")
        if folder:
            self.output_folder = folder
            self.lbl_out.setText(folder)

    def get_operations(self):
        ops = []
        if self.chk_gray.isChecked():
            ops.append((preprocessing.rgb_to_gray, []))
        if self.chk_binary.isChecked():
            ops.append((preprocessing.gray_to_binary, [self.spin_binary.value()]))
        if self.chk_hist.isChecked():
            ops.append((enhancement.histogram_stretching, []))
        if self.chk_contrast.isChecked():
            ops.append((enhancement.contrast_enhancement, [self.spin_contrast.value()]))
        if self.chk_median.isChecked():
            ops.append((filtering.median_filter, [int(self.combo_median.currentText())]))
        if self.chk_unsharp.isChecked():
            ops.append((filtering.unsharp_mask, [self.spin_unsharp.value()]))
        if self.chk_prewitt.isChecked():
            ops.append((analysis.prewitt_edge_detection, []))
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
        self.history = []
        self.setup_ui()
        self.view_zoom_factor = 1.0
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._apply_slider_preview)
        self.current_preview_op = None
        self._resize_fit_timer = QTimer(self)
        self._resize_fit_timer.setSingleShot(True)
        self._resize_fit_timer.timeout.connect(self._refit_image_panels)
        _qs = QShortcut(QKeySequence(Qt.Key_Q), self)
        _qs.setContext(Qt.WindowShortcut)
        _qs.activated.connect(self.close)

    def setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)
        ml.setSpacing(8)
        ml.setContentsMargins(8, 6, 8, 8)
        top_h = QHBoxLayout()
        lc = QVBoxLayout()
        self.lbl_orig_meta = QLabel()
        self.lbl_orig_meta.setObjectName("panelMeta")
        self.lbl_orig_meta.setTextFormat(Qt.RichText)
        self.lbl_orig_meta.setWordWrap(True)
        self.lbl_orig_meta.setMinimumHeight(34)
        self.left_label = QLabel(
            "<div align='center'><span style='font-size:15px;font-weight:600;color:#ffffff'>Kaynak görüntü</span><br/>"
            "<span style='font-size:12px;color:#dce0eb'>Araç çubuğundan « Görüntü Yükle »</span></div>"
        )
        self.left_label.setTextFormat(Qt.RichText)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setObjectName("panelImage")
        self.left_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.left_label.setMinimumSize(120, 80)
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(False)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.left_scroll.setMinimumSize(180, 160)
        # İçteki pixmap boyutunda minimum istenmesin; yoksa üst panel tüm pencereyi kaplar, alt menü sıkışır.
        self.left_scroll.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.left_scroll.setWidget(self.left_label)
        lc.addWidget(self.lbl_orig_meta)
        lc.addWidget(self.left_scroll, 1)
        lw = QWidget()
        lw.setLayout(lc)
        top_h.addWidget(lw, 2)
        rc = QVBoxLayout()
        self.lbl_proc_meta = QLabel()
        self.lbl_proc_meta.setObjectName("panelMeta")
        self.lbl_proc_meta.setTextFormat(Qt.RichText)
        self.lbl_proc_meta.setWordWrap(True)
        self.lbl_proc_meta.setMinimumHeight(34)
        self.right_label = QLabel(
            "<div align='center'><span style='font-size:15px;font-weight:600;color:#ffffff'>İşlenmiş önizleme</span><br/>"
            "<span style='font-size:12px;color:#dce0eb'>İşlemler bu panelde</span></div>"
        )
        self.right_label.setTextFormat(Qt.RichText)
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setObjectName("panelImage")
        self.right_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.right_label.setMinimumSize(120, 80)
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(False)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setMinimumSize(180, 160)
        self.right_scroll.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.right_scroll.setWidget(self.right_label)
        self.right_scroll.viewport().installEventFilter(self)
        rc.addWidget(self.lbl_proc_meta)
        rc.addWidget(self.right_scroll, 1)
        rw = QWidget()
        rw.setLayout(rc)
        top_h.addWidget(rw, 2)
        hg = QGroupBox("İşlem Geçmişi")
        hg.setObjectName("historyCard")
        hl = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.undo_operation_from_list)
        hl.addWidget(self.history_list)
        hb = QHBoxLayout()
        self.btn_undo = QPushButton("Son İşlemi Geri Al (Undo)")
        self.btn_undo.clicked.connect(self.undo_operation)
        hb.addWidget(self.btn_undo)
        hl.addLayout(hb)
        hg.setLayout(hl)
        top_h.addWidget(hg, 1)
        top_inner = QWidget()
        til = QVBoxLayout(top_inner)
        til.setContentsMargins(0, 0, 0, 0)
        til.addLayout(top_h)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(350)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sp = QSplitter(Qt.Vertical)
        sp.addWidget(top_inner)
        sp.addWidget(self.tabs)
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 0)
        sp.setSizes([450, 350])
        sp.setChildrenCollapsible(False)
        ml.addWidget(sp, 1)
        self.setup_tabs()
        self.setup_toolbar()
        self.update_panel_metadata()

    def setup_toolbar(self):
        tb = QToolBar("Ana Araç Çubuğu")
        tb.setMovable(False)
        self.addToolBar(tb)
        al = QAction("Görüntü Yükle", self)
        al.triggered.connect(self.load_image)
        tb.addAction(al)
        as_ = QAction("Kaydet", self)
        as_.triggered.connect(self.save_image)
        tb.addAction(as_)
        ar = QAction("Sıfırla", self)
        ar.triggered.connect(self.reset_image)
        tb.addAction(ar)
        ab = QAction("Toplu İşlem", self)
        ab.triggered.connect(self.batch_process)
        tb.addAction(ab)

    def setup_tabs(self):
        t1 = QWidget()
        t1.setObjectName("tabPage")
        t1.setAttribute(Qt.WA_StyledBackground, True)
        l1 = QVBoxLayout(t1)
        l1.setContentsMargins(2, 2, 2, 2)
        g1 = QGroupBox("Temel dönüşüm")
        v1 = QVBoxLayout()
        b1 = QPushButton("Gri Dönüşüm")
        b1.clicked.connect(lambda: self.apply_operation("Modül A: Gri Forma Dönüştürüldü", preprocessing.rgb_to_gray))
        v1.addWidget(b1)
        g1.setLayout(v1)
        g2 = QGroupBox("Binary")
        v2 = QVBoxLayout()
        h2 = QHBoxLayout()
        self.slider_binary = QSlider(Qt.Horizontal)
        self.slider_binary.setRange(0, 255)
        self.slider_binary.setValue(128)
        self.label_binary_val = QLabel("128")
        self.label_binary_val.setMinimumWidth(35)
        self.label_binary_val.setAlignment(Qt.AlignCenter)
        self.slider_binary.valueChanged.connect(lambda v: self.label_binary_val.setText(str(v)))
        bb = QPushButton("Binary Dönüşüm")
        bb.clicked.connect(
            lambda: self.apply_operation(
                f"Modül A: Siyah Beyaz (Binary) Dönüşüm (Eşik {self.slider_binary.value()})",
                preprocessing.gray_to_binary,
                self.slider_binary.value(),
            )
        )
        h2.addWidget(self.slider_binary)
        h2.addWidget(self.label_binary_val)
        h2.addWidget(bb)
        v2.addLayout(h2)
        g2.setLayout(v2)
        g3 = QGroupBox("HSV")
        v3 = QVBoxLayout()
        bh = QPushButton("HSV Dönüşüm")
        bh.clicked.connect(lambda: self.apply_operation("Modül A: Renk Uzayı HSV'ye Dönüştürüldü", preprocessing.rgb_to_hsv))
        v3.addWidget(bh)
        g3.setLayout(v3)
        r1 = QHBoxLayout()
        r1.addWidget(g1, 1)
        r1.addWidget(g2, 1)
        r1.addWidget(g3, 1)
        l1.addLayout(r1)
        self.tabs.addTab(t1, "Ön İşleme (Modül A)")

        t2 = QWidget()
        t2.setObjectName("tabPage")
        t2.setAttribute(Qt.WA_StyledBackground, True)
        l2 = QVBoxLayout(t2)
        l2.setContentsMargins(2, 2, 2, 2)
        gr = QGroupBox("Döndürme")
        lr = QHBoxLayout()
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.spin_angle.setValue(45)
        br = QPushButton("Döndür")
        br.clicked.connect(
            lambda: self.apply_operation(
                f"Modül B: Ters Haritalama ile Döndürme ({self.spin_angle.value()}°)",
                geometry.rotate_image,
                float(self.spin_angle.value()),
            )
        )
        lr.addWidget(self.spin_angle)
        lr.addWidget(br)
        gr.setLayout(lr)
        gc = QGroupBox("Kırpma")
        lc = QHBoxLayout()
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 9999)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 9999)
        self.spin_w = QSpinBox()
        self.spin_w.setRange(1, 9999)
        self.spin_w.setValue(200)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(1, 9999)
        self.spin_h.setValue(200)
        bc = QPushButton("Kırp")
        bc.clicked.connect(
            lambda: self.apply_operation(
                f"Modül B: Kırpma İşlemi ({self.spin_w.value()}x{self.spin_h.value()})",
                geometry.crop_image,
                self.spin_x.value(),
                self.spin_y.value(),
                self.spin_w.value(),
                self.spin_h.value(),
            )
        )
        for w in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            lc.addWidget(w)
        lc.addWidget(bc)
        gc.setLayout(lc)
        gz = QGroupBox("Zoom")
        lz = QVBoxLayout()
        lz_slider = QHBoxLayout()
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(1, 40)
        self.slider_zoom.setValue(10)
        self.label_zoom = QLabel("1.0x")
        self.label_zoom.setMinimumWidth(45)
        self.label_zoom.setAlignment(Qt.AlignCenter)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)
        bz = QPushButton("Zoom Uygula")
        bz.clicked.connect(
            lambda: self.apply_operation(
                f"Modül B: Bilineer İnterpolasyon ile Zoom ({self.slider_zoom.value()/10.0}x)",
                geometry.zoom_image,
                self.slider_zoom.value() / 10.0,
            )
        )
        lz_slider.addWidget(self.slider_zoom)
        lz_slider.addWidget(self.label_zoom)
        lz.addLayout(lz_slider)
        lz.addWidget(bz)
        gz.setLayout(lz)
        l2.addWidget(gr)
        l2.addWidget(gc)
        l2.addWidget(gz)
        self.tabs.addTab(t2, "Geometrik Düzeltme (Modül B)")

        t3 = QWidget()
        t3.setObjectName("tabPage")
        t3.setAttribute(Qt.WA_StyledBackground, True)
        l3 = QVBoxLayout(t3)
        l3.setContentsMargins(2, 2, 2, 2)
        gh = QGroupBox("Histogram")
        vh = QVBoxLayout()
        bh_show = QPushButton("Histogram Analizini Göster")
        bh_show.clicked.connect(self.show_histogram_dialog)
        bh_st = QPushButton("Histogram Germe")
        bh_st.clicked.connect(self._apply_histogram_stretch)
        vh.addWidget(bh_show)
        vh.addWidget(bh_st)
        gh.setLayout(vh)
        gt = QGroupBox("Kontrast / Gamma")
        ht = QHBoxLayout()
        
        # Kontrast
        vk = QVBoxLayout()
        hk = QHBoxLayout()
        self.slider_cont = QSlider(Qt.Horizontal)
        self.slider_cont.setRange(1, 30)
        self.slider_cont.setValue(10)
        self.label_cont = QLabel("1.0x")
        self.label_cont.setMinimumWidth(45)
        self.label_cont.setAlignment(Qt.AlignCenter)
        self.slider_cont.valueChanged.connect(self._on_contrast_changed)
        bc2 = QPushButton("Kontrast Artır")
        bc2.clicked.connect(
            lambda: self.apply_operation(
                f"Modül C: Kontrast Artırıldı ({self.slider_cont.value()/10.0}x)",
                enhancement.contrast_enhancement,
                self.slider_cont.value() / 10.0,
            )
        )
        hk.addWidget(self.slider_cont)
        hk.addWidget(self.label_cont)
        vk.addLayout(hk)
        vk.addWidget(bc2)
        ht.addLayout(vk)

        # Gamma
        vg = QVBoxLayout()
        hg = QHBoxLayout()
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(1, 50)
        self.slider_gamma.setValue(10)
        self.label_gamma = QLabel("γ=1.0")
        self.label_gamma.setMinimumWidth(45)
        self.label_gamma.setAlignment(Qt.AlignCenter)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        bg = QPushButton("Gamma")
        bg.clicked.connect(
            lambda: self.apply_operation(
                f"Modül C: Gamma Düzeltmesi (γ={self.slider_gamma.value()/10.0:.1f})",
                enhancement.gamma_correction,
                self.slider_gamma.value() / 10.0,
            )
        )
        hg.addWidget(self.slider_gamma)
        hg.addWidget(self.label_gamma)
        vg.addLayout(hg)
        vg.addWidget(bg)
        ht.addLayout(vg)

        gt.setLayout(ht)
        gn = QGroupBox("Gürültü")
        vn = QVBoxLayout()
        vn_slider = QHBoxLayout()
        self.slider_sp = QSlider(Qt.Horizontal)
        self.slider_sp.setRange(1, 100)
        self.slider_sp.setValue(5)
        self.label_sp = QLabel("%5")
        self.label_sp.setMinimumWidth(45)
        self.label_sp.setAlignment(Qt.AlignCenter)
        self.slider_sp.valueChanged.connect(self._on_noise_changed)
        bn = QPushButton("Tuz/Biber")
        bn.clicked.connect(
            lambda: self.apply_operation(
                f"Modül C: Tuz/Biber Gürültüsü Eklendi (%{self.slider_sp.value()})",
                enhancement.add_salt_pepper_noise,
                float(self.slider_sp.value()),
            )
        )
        vn_slider.addWidget(self.slider_sp)
        vn_slider.addWidget(self.label_sp)
        vn.addLayout(vn_slider)
        vn.addWidget(bn)
        gn.setLayout(vn)
        ga = QGroupBox("İki görüntü")
        va = QHBoxLayout()
        ba = QPushButton("Ortalama (2. dosya)")
        ba.clicked.connect(self.do_image_averaging)
        ba2 = QPushButton("Ekleme")
        ba2.clicked.connect(self.do_image_addition)
        ba3 = QPushButton("Bölme")
        ba3.clicked.connect(self.do_image_division)
        va.addWidget(ba)
        va.addWidget(ba2)
        va.addWidget(ba3)
        ga.setLayout(va)
        r3 = QHBoxLayout()
        r3.addWidget(gh, 1)
        r3.addWidget(gt, 1)
        l3.addLayout(r3)
        r3_bottom = QHBoxLayout()
        r3_bottom.addWidget(gn, 1)
        r3_bottom.addWidget(ga, 1)
        l3.addLayout(r3_bottom)
        self.tabs.addTab(t3, "İstatistiksel İyileştirme (Modül C)")

        t4 = QWidget()
        t4.setObjectName("tabPage")
        t4.setAttribute(Qt.WA_StyledBackground, True)
        l4 = QVBoxLayout(t4)
        l4.setContentsMargins(2, 2, 2, 2)
        gm = QGroupBox("Mean")
        mm = QHBoxLayout()
        self.spin_mean_k = QSpinBox()
        self.spin_mean_k.setRange(3, 15)
        self.spin_mean_k.setSingleStep(2)
        self.spin_mean_k.setValue(3)
        bm = QPushButton("Mean")
        bm.clicked.connect(
            lambda: self.apply_operation(f"Modül D: Mean Filtresi ({self.spin_mean_k.value()}x{self.spin_mean_k.value()})", filtering.mean_filter, self.spin_mean_k.value())
        )
        mm.addWidget(self.spin_mean_k)
        mm.addWidget(bm)
        gm.setLayout(mm)
        gmed = QGroupBox("Median")
        mmed = QHBoxLayout()
        self.combo_median_main = QComboBox()
        self.combo_median_main.addItems(["3", "5", "7", "9"])
        bmed = QPushButton("Median")
        bmed.clicked.connect(
            lambda: self.apply_operation(
                f"Modül D: Median Filtresi ({self.combo_median_main.currentText()}x{self.combo_median_main.currentText()})",
                filtering.median_filter,
                int(self.combo_median_main.currentText()),
            )
        )
        mmed.addWidget(self.combo_median_main)
        mmed.addWidget(bmed)
        gmed.setLayout(mmed)
        gg = QGroupBox("Gaussian")
        mg = QHBoxLayout()
        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.5, 4.0)
        self.spin_gauss_sigma.setValue(1.0)
        bg2 = QPushButton("Gaussian")
        bg2.clicked.connect(
            lambda: self.apply_operation(f"Modül D: Gaussian Bulanıklık (σ={self.spin_gauss_sigma.value()})", gaussian_blur_sigma, self.spin_gauss_sigma.value())
        )
        mg.addWidget(self.spin_gauss_sigma)
        mg.addWidget(bg2)
        gg.setLayout(mg)
        gu = QGroupBox("Unsharp")
        mu = QHBoxLayout()
        self.spin_unsharp_main = QDoubleSpinBox()
        self.spin_unsharp_main.setRange(0.1, 5.0)
        self.spin_unsharp_main.setValue(1.5)
        bu = QPushButton("Unsharp")
        bu.clicked.connect(
            lambda: self.apply_operation(f"Modül D: Unsharp Masking Keskinleştirme ({self.spin_unsharp_main.value()})", filtering.unsharp_mask, self.spin_unsharp_main.value())
        )
        mu.addWidget(self.spin_unsharp_main)
        mu.addWidget(bu)
        gu.setLayout(mu)
        r4a = QHBoxLayout()
        r4a.addWidget(gm, 1)
        r4a.addWidget(gmed, 1)
        r4b = QHBoxLayout()
        r4b.addWidget(gg, 1)
        r4b.addWidget(gu, 1)
        l4.addLayout(r4a)
        l4.addLayout(r4b)
        self.tabs.addTab(t4, "Filtreleme (Modül D)")

        t5 = QWidget()
        t5.setObjectName("tabPage")
        t5.setAttribute(Qt.WA_StyledBackground, True)
        l5 = QVBoxLayout(t5)
        l5.setContentsMargins(2, 2, 2, 2)
        ge = QGroupBox("Kenar")
        ve = QVBoxLayout()
        bp = QPushButton("Prewitt")
        bp.clicked.connect(lambda: self.apply_operation("Modül E: Prewitt ile Kenar Bulma", analysis.prewitt_edge_detection))
        ve.addWidget(bp)
        ge.setLayout(ve)
        gmo = QGroupBox("Morfoloji")
        self.combo_morph_shape = QComboBox()
        self.combo_morph_shape.addItems(["rect", "cross", "ellipse"])
        self.combo_morph = QComboBox()
        for s in (3, 5, 7, 9):
            self.combo_morph.addItem(str(s))
        vm = QVBoxLayout()
        hm = QHBoxLayout()
        hm.addWidget(QLabel("Şekil"))
        hm.addWidget(self.combo_morph_shape)
        hm.addWidget(QLabel("Boyut"))
        hm.addWidget(self.combo_morph)
        vm.addLayout(hm)
        h2 = QHBoxLayout()
        for txt, fn in (
            ("Erozyon", analysis.erosion),
            ("Dilation", analysis.dilation),
            ("Opening", analysis.opening),
            ("Closing", analysis.closing),
        ):
            b = QPushButton(txt)
            b.clicked.connect(
                lambda checked=False, f=fn, name=f"Modül E: Morfolojik İşlem ({txt})": self.apply_operation(
                    name, f, self._morph_structuring_element(), "auto"
                )
            )
            h2.addWidget(b)
        vm.addLayout(h2)
        gmo.setLayout(vm)
        r5 = QHBoxLayout()
        r5.addWidget(ge, 1)
        r5.addWidget(gmo, 2)
        l5.addLayout(r5)
        self.tabs.addTab(t5, "Morfoloji (Modül E)")

    def show_histogram_dialog(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Hata", "Lütfen önce görüntü yükleyin!")
            return
        try:
            hist = enhancement.compute_histogram(self.current_image)
            d = QDialog(self)
            d.setWindowTitle("Histogram Analizi")
            dl = QVBoxLayout(d)
            fig = Figure(figsize=(7, 4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.bar(range(256), hist, color="#333", width=1.0)
            ax.set_title("Histogram Analizi")
            ax.set_xlabel("Yoğunluk")
            ax.set_ylabel("Frekans")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            canvas.draw()
            dl.addWidget(canvas)
            h = QHBoxLayout()
            h.addStretch(1)
            bc = QPushButton("Kapat")
            bc.clicked.connect(d.accept)
            h.addWidget(bc)
            dl.addLayout(h)
            d.resize(780, 500)
            d.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Histogram Hatası", str(e))

    def update_panel_metadata(self):
        def ch(img):
            if img is None:
                return "—"
            if len(img.shape) == 3:
                return f"{img.shape[2]} kanal (BGR)"
            return "Tek kanal"

        def card(title, lines):
            esc = "<br/>".join(html.escape(x) for x in lines)
            return (
                f"<div><span style='font-weight:700;color:#ffffff'>{html.escape(title)}</span>"
                f"<span style='display:block;margin-top:6px;color:#e2e8f0;font-size:12px'>{esc}</span></div>"
            )

        if self.original_image is None:
            self.lbl_orig_meta.setText(
                card("Kaynak", ["Dosya: —", "Çözünürlük: —", "Kanallar: —"])
            )
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "—"
            ho, wo = self.original_image.shape[:2]
            self.lbl_orig_meta.setText(
                card("Kaynak", [f"Dosya: {fn}", f"Çözünürlük: {wo}×{ho}", f"Kanal: {ch(self.original_image)}"])
            )
        if self.current_image is None:
            self.lbl_proc_meta.setText(card("İşlenmiş", ["—", "—", "—"]))
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "—"
            hc, wc = self.current_image.shape[:2]
            last = self.history[-1][0] if self.history else "—"
            self.lbl_proc_meta.setText(
                card(
                    "İşlenmiş",
                    [f"Dosya: {fn}", f"Çözünürlük: {wc}×{hc}", f"Kanal: {ch(self.current_image)}", f"Son: {last}"],
                )
            )

    def _morph_structuring_element(self):
        return analysis.get_structuring_element(
            self.combo_morph_shape.currentText(),
            int(self.combo_morph.currentText()),
        )

    def _apply_histogram_stretch(self):
        self.apply_operation("Modül C: Histogram Germe uygulandı", enhancement.histogram_stretching)

    def _should_fit_display(self, img_array):
        """Sağ panel: yalnızca geçmişteki son işlem Zoom ise tam piksel; diğer tüm durullarda panele sığdır."""
        if img_array is None:
            return True
        if not self.history:
            return True
        return not self.history[-1][0].startswith("Zoom ")

    def apply_operation(self, op_name, module_func, *args):
        if self.current_image is None:
            QMessageBox.warning(self, "Hata", "Önce görüntü yükleyin!")
            return
        # İşleve bağımsız kopya ver: girdi üzerinde yazan fonksiyonlar current_image'ı bozmasın; prev geri al için sabit kalır.
        prev = np.copy(self.current_image)
        try:
            result = module_func(np.copy(prev), *args)
        except NotImplementedError as e:
            QMessageBox.information(self, "Bilgi", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "İşlem Hatası", f"«{op_name}»\n{str(e)}")
            return
        if result is None:
            return
        hn = len(self.history)
        lr = self.history_list.count()
        try:
            self.history.append((op_name, module_func, args, prev))
            self.history_list.addItem(f"{len(self.history)}. {op_name}")
            self.update_image_display(result, self.right_label, self.right_scroll)
            self.current_image = result
            self.update_panel_metadata()
        except Exception as e:
            self.current_image = prev
            self.history = self.history[:hn]
            while self.history_list.count() > lr:
                self.history_list.takeItem(self.history_list.count() - 1)
            try:
                self.update_image_display(prev, self.right_label, self.right_scroll)
            except Exception:
                pass
            self.update_panel_metadata()
            QMessageBox.critical(self, "Güncelleme Hatası", f"«{op_name}»\n{str(e)}")

    def undo_operation_from_list(self, item):
        self.restore_history_state(self.history_list.row(item))

    def undo_operation(self):
        if self.history:
            self.restore_history_state(len(self.history) - 1)

    def restore_history_state(self, row_index):
        if row_index < 0 or row_index >= len(self.history):
            return
        prev_state = self.history[row_index][3]
        self.history = self.history[:row_index]
        while self.history_list.count() > row_index:
            self.history_list.takeItem(row_index)
        self.current_image = np.copy(prev_state)
        self.update_image_display(self.current_image, self.right_label, self.right_scroll)
        self.update_panel_metadata()

    def do_image_averaging(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "2. görüntü", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if not fn:
            return
        img2 = safe_imread(fn)
        if img2 is None:
            QMessageBox.warning(self, "Hata", "Dosya okunamadı")
            return
        if img2.shape != self.current_image.shape:
            QMessageBox.warning(
                self,
                "Boyut uyuşmazlığı",
                f"Ana: {self.current_image.shape}, seçilen: {img2.shape}",
            )
            return
        self.apply_operation("Modül C: Görüntü Ortalama (Averaging)", enhancement.image_averaging, img2)

    def do_image_addition(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "2. görüntü", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            img2 = safe_imread(fn)
            if img2 is not None:
                self.apply_operation("Modül C: Görüntüler Toplandı", geometry.add_images, img2)

    def do_image_division(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "2. görüntü", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            img2 = safe_imread(fn)
            if img2 is not None:
                self.apply_operation("Modül C: Görüntüler Bölündü", geometry.divide_images, img2)

    def load_image(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Görüntü Seç", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            self.original_image = safe_imread(fn)
            if self.original_image is not None:
                self.image_path = fn
                self.current_image = self.original_image.copy()
                self.history.clear()
                self.history_list.clear()
                self.update_image_display(self.original_image, self.left_label, self.left_scroll)
                self.update_image_display(self.current_image, self.right_label, self.right_scroll)
                self.update_panel_metadata()
            else:
                self.image_path = None
                QMessageBox.warning(self, "Hata", "Görüntü okunamadı")

    def save_image(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Kaydet", "islenmis.jpg", "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"
        )
        if fn:
            safe_imwrite(fn, self.current_image)
            QMessageBox.information(self, "Tamam", "Kaydedildi")

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.history.clear()
            self.history_list.clear()
            self.update_image_display(self.current_image, self.right_label, self.right_scroll)
            self.update_panel_metadata()

    def batch_process(self):
        dlg = BatchProcessDialog(self)
        if not dlg.exec_():
            return
        if not dlg.input_files or not dlg.output_folder:
            QMessageBox.warning(self, "Uyarı", "Girdi ve çıktı seçin")
            return
        ops = dlg.get_operations()
        if not ops:
            QMessageBox.warning(self, "Uyarı", "En az bir filtre seçin")
            return
        try:
            n = 0
            last = None
            for fp in dlg.input_files:
                orig = safe_imread(fp)
                if orig is None:
                    continue
                self.update_image_display(orig, self.left_label, self.left_scroll)
                QApplication.processEvents()
                img = orig.copy()
                for f, a in ops:
                    r = f(img, *a)
                    if r is not None:
                        img = r
                safe_imwrite(os.path.join(dlg.output_folder, os.path.basename(fp)), img)
                n += 1
                last = img
                self.update_image_display(img, self.right_label, self.right_scroll)
                QApplication.processEvents()
            QMessageBox.information(self, "Batch", f"{n} dosya işlendi")
            if last is not None:
                pd = QDialog(self)
                pd.setWindowTitle("Önizleme")
                pl = QVBoxLayout(pd)
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                sc = QScrollArea()
                sc.setWidgetResizable(False)
                sc.setWidget(lbl)
                self.update_image_display(last, lbl, sc)
                pl.addWidget(sc)
                pb = QPushButton("Kapat")
                pb.clicked.connect(pd.accept)
                pl.addWidget(pb)
                pd.resize(700, 700)
                pd.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Batch Hata", str(e))

    def _numpy_to_qpixmap(self, img_u8):
        """uint8 BGR veya grayscale numpy → QPixmap (fromImage içerik kopyalar)."""
        h, w = img_u8.shape[:2]
        if len(img_u8.shape) == 3:
            rgb = img_u8[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            g = np.ascontiguousarray(img_u8)
            qimg = QImage(g.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def _on_contrast_changed(self, v):
        self.label_cont.setText(f"{v/10.0}x")
        self.current_preview_op = (enhancement.contrast_enhancement, v / 10.0)
        self.debounce_timer.start(150)

    def _on_gamma_changed(self, v):
        self.label_gamma.setText(f"γ={v/10.0:.1f}")
        self.current_preview_op = (enhancement.gamma_correction, v / 10.0)
        self.debounce_timer.start(150)

    def _on_noise_changed(self, v):
        self.label_sp.setText(f"%{v}")
        self.current_preview_op = (enhancement.add_salt_pepper_noise, float(v))
        self.debounce_timer.start(150)

    def _on_zoom_changed(self, v):
        self.label_zoom.setText(f"{v/10.0}x")
        self.current_preview_op = (self._fast_zoom_preview, v / 10.0)
        self.debounce_timer.start(150)

    def _fast_zoom_preview(self, img, scale):
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return self.manual_nearest_neighbor(img, new_w, new_h)

    def _apply_slider_preview(self):
        if self.current_preview_op and self.current_image is not None:
            func, val = self.current_preview_op
            # Elif: Slider kaydırılırken anlık önizleme için debounce mekanizması ekledik ki arayüz donmasın.
            try:
                preview = func(np.copy(self.current_image), val)
                self.update_image_display(preview, self.right_label, self.right_scroll)
            except Exception:
                pass

    def manual_nearest_neighbor(self, image, new_w, new_h):
        # Nazlı: Pikselleri yeni boyutlara göre eşleştiriyoruz (Nearest Neighbor).
        # Arayüz tepkimesinin yavaşlamaması için numpy vektörizasyonu ile hızlandırdık.
        h, w = image.shape[:2]
        if new_w == 0 or new_h == 0:
            return image
        y_ratio = h / float(new_h)
        x_ratio = w / float(new_w)
        y_indices = (np.arange(new_h) * y_ratio).astype(int)
        x_indices = (np.arange(new_w) * x_ratio).astype(int)
        return image[y_indices[:, None], x_indices]

    def eventFilter(self, source, event):
        if source is self.right_scroll.viewport() and event.type() == event.Wheel:
            if self.current_image is not None:
                # Bengü: Yakınlaştırma sırasında görüntünün merkezini farenin olduğu koordinata sabitleyerek kaymayı engelledik.
                angle = event.angleDelta().y()
                zoom_step = 1.1 if angle > 0 else 0.9
                
                old_factor = self.view_zoom_factor
                self.view_zoom_factor *= zoom_step
                self.view_zoom_factor = max(0.1, min(self.view_zoom_factor, 10.0))
                
                if old_factor != self.view_zoom_factor:
                    mouse_pos = event.pos()
                    old_x = self.right_scroll.horizontalScrollBar().value() + mouse_pos.x()
                    old_y = self.right_scroll.verticalScrollBar().value() + mouse_pos.y()
                    
                    h, w = self.current_image.shape[:2]
                    new_w = max(1, int(w * self.view_zoom_factor))
                    new_h = max(1, int(h * self.view_zoom_factor))
                    
                    zoomed_img = self.manual_nearest_neighbor(self.current_image, new_w, new_h)
                    pm = self._numpy_to_qpixmap(zoomed_img)
                    self.right_label.setPixmap(pm)
                    self.right_label.adjustSize()
                    
                    ratio = self.view_zoom_factor / old_factor
                    new_x = int(old_x * ratio) - mouse_pos.x()
                    new_y = int(old_y * ratio) - mouse_pos.y()
                    self.right_scroll.horizontalScrollBar().setValue(new_x)
                    self.right_scroll.verticalScrollBar().setValue(new_y)
                return True
        return super().eventFilter(source, event)

    def update_image_display(self, img_array, label, scroll_area=None, fit_viewport=None):
        if img_array is None:
            return
        if fit_viewport is None:
            if label is self.left_label:
                fit_viewport = True
            elif label is self.right_label:
                fit_viewport = self._should_fit_display(img_array)
            else:
                fit_viewport = True
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        h, w = img_array.shape[:2]
        label.setText("")
        label.setScaledContents(False)

        if scroll_area is not None:
            vp = scroll_area.viewport()
            if vp.width() < 8 or vp.height() < 8:

                def _refit():
                    if label is self.left_label and self.original_image is not None:
                        self.update_image_display(
                            self.original_image, self.left_label, self.left_scroll
                        )
                    elif label is self.right_label and self.current_image is not None:
                        self.update_image_display(
                            self.current_image, self.right_label, self.right_scroll
                        )
                    elif scroll_area.widget() is label:
                        self.update_image_display(img_array, label, scroll_area)

                QTimer.singleShot(0, _refit)
                return

            if fit_viewport:
                vw = max(1, vp.width() - 2)
                vh = max(1, vp.height() - 2)
                scale = min(vw / w, vh / h, 1.0)
                rw = max(1, int(round(w * scale)))
                rh = max(1, int(round(h * scale)))
                if (rw, rh) != (w, h):
                    # Resmi sığdırmak için kendi yazdığımız nearest neighbor interpolasyon fonksiyonunu kullanıyoruz.
                    img_array = self.manual_nearest_neighbor(img_array, rw, rh)
        pm_show = self._numpy_to_qpixmap(img_array)
        label.setPixmap(pm_show)
        label.adjustSize()

    def _refit_image_panels(self):
        try:
            if self.original_image is not None:
                self.update_image_display(
                    self.original_image, self.left_label, self.left_scroll
                )
            if self.current_image is not None:
                self.update_image_display(
                    self.current_image, self.right_label, self.right_scroll
                )
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_fit_timer.start(120)


MODERN_STYLE = """
QMainWindow, QDialog { background-color: #14151c; }
QWidget { color: #ececf3; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QLabel { color: #dce0eb; }
QLabel#panelMeta { padding: 10px 12px; background: #1e2030; border: 1px solid #2e3348; border-radius: 12px; }
QLabel#panelImage { border: 1px dashed #3d4a63; border-radius: 14px; background: #1a1c28; padding: 8px; }
QGroupBox#historyCard { border: 1px solid #2e3348; border-radius: 14px; margin-top: 8px; padding: 10px; background: #1e2030; }
QWidget#tabPage { background: #181a26; }
QGroupBox { border: 1px solid #323848; border-radius: 10px; margin-top: 6px; padding: 8px; background: #222534; }
QPushButton { background: #3b4256; border: 1px solid #4a5568; border-radius: 8px; padding: 12px 14px; color: #ffffff; font-weight: 500; min-height: 42px; text-align: center; }
QPushButton:hover { background: #4a5568; border-color: #5d8cff; }
QTabWidget::pane { border: 1px solid #2e3348; border-radius: 12px; background: #181a26; padding: 6px; }
QTabBar::tab { background: #1e2230; color: #8b92a8; padding: 8px 16px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
QTabBar::tab:selected { background: #181a26; color: #ffffff; font-weight: 600; border-bottom: 3px solid #5d8cff; }
QSlider::groove:horizontal { height: 6px; background: #12141d; border-radius: 3px; }
QSlider::handle:horizontal { background: #e8ebf4; border: 1px solid #5d8cff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QListWidget { background: #151721; border: 1px solid #2e3348; border-radius: 12px; }
QSpinBox, QDoubleSpinBox, QComboBox { background: #222534; border: 1px solid #3d4660; border-radius: 8px; padding: 4px 8px; color: #ececf3; }
QToolBar { background: #1a1c28; border-bottom: 1px solid #2e3348; padding: 8px; }
QScrollBar:vertical { background: #151721; width: 10px; }
QScrollBar::handle:vertical { background: #3d455c; border-radius: 5px; min-height: 24px; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(MODERN_STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
