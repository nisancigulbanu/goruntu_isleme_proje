import html
import os

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .batch_dialog import BatchProcessDialog
from .image_io import safe_imread, safe_imwrite
from .styles import build_modern_style


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processing_modules = {
            "preprocessing": preprocessing,
            "geometry": geometry,
            "enhancement": enhancement,
            "filtering": filtering,
            "analysis": analysis,
        }
        self.setWindowTitle("Dijital Arşivci - Tarihi Belge ve Fotoğraf Restorasyonu")
        self.resize(1150, 820)
        self.setMinimumSize(860, 620)
        self._ui_scale = 1.0
        self.original_image = None
        self.current_image = None
        self.image_path = None
        self.history = []
        self.setup_ui()
        self._resize_fit_timer = QTimer(self)
        self._resize_fit_timer.setSingleShot(True)
        self._resize_fit_timer.timeout.connect(self._refit_image_panels)
        _qs = QShortcut(QKeySequence(Qt.Key_Q), self)
        _qs.setContext(Qt.WindowShortcut)
        _qs.activated.connect(self.close)

    def _calc_ui_scale(self):
        width_scale = self.width() / 1280.0
        height_scale = self.height() / 860.0
        return min(1.45, max(1.0, min(width_scale, height_scale)))

    def _scale_px(self, value):
        return max(1, int(round(value * self._ui_scale)))

    def _panel_placeholder_html(self, title, subtitle):
        title_size = self._scale_px(18)
        subtitle_size = self._scale_px(14)
        return (
            f"<div align='center'><span style='font-size:{title_size}px;font-weight:600;color:#9aa3b8'>"
            f"{html.escape(title)}</span><br/>"
            f"<span style='font-size:{subtitle_size}px;color:#5f677a'>{html.escape(subtitle)}</span></div>"
        )

    def _apply_responsive_ui(self, force=False):
        scale = self._calc_ui_scale()
        if not force and abs(scale - self._ui_scale) < 0.04:
            return

        self._ui_scale = scale
        self.setStyleSheet(build_modern_style(scale))
        self.lbl_orig_meta.setMinimumHeight(self._scale_px(72))
        self.lbl_proc_meta.setMinimumHeight(self._scale_px(72))
        self.left_scroll.setMinimumSize(self._scale_px(220), self._scale_px(190))
        self.right_scroll.setMinimumSize(self._scale_px(220), self._scale_px(190))
        self.tabs.setMinimumHeight(self._scale_px(340))

        if self.original_image is None:
            self.left_label.setText(
                self._panel_placeholder_html(
                    "Kaynak görüntü",
                    "Araç çubuğundan « Görüntü Yükle »",
                )
            )
        if self.current_image is None:
            self.right_label.setText(
                self._panel_placeholder_html(
                    "İşlenmiş önizleme",
                    "İşlemler bu panelde",
                )
            )

        total_height = max(self.height(), self.minimumHeight())
        self.main_splitter.setSizes([int(total_height * 0.57), int(total_height * 0.43)])
        self.update_panel_metadata()

    def setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)
        ml.setSpacing(10)
        ml.setContentsMargins(10, 8, 10, 10)

        top_h = QHBoxLayout()
        top_h.setSpacing(12)

        lc = QVBoxLayout()
        lc.setSpacing(10)
        self.lbl_orig_meta = QLabel()
        self.lbl_orig_meta.setObjectName("panelMeta")
        self.lbl_orig_meta.setTextFormat(Qt.RichText)
        self.lbl_orig_meta.setWordWrap(True)
        self.lbl_orig_meta.setMinimumHeight(72)
        self.left_label = QLabel(
            "<div align='center'><span style='font-size:15px;font-weight:600;color:#9aa3b8'>Kaynak görüntü</span><br/>"
            "<span style='font-size:12px;color:#5f677a'>Araç çubuğundan « Görüntü Yükle »</span></div>"
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
        self.left_scroll.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.left_scroll.setWidget(self.left_label)
        lc.addWidget(self.lbl_orig_meta)
        lc.addWidget(self.left_scroll, 1)
        lw = QWidget()
        lw.setLayout(lc)
        top_h.addWidget(lw, 2)

        rc = QVBoxLayout()
        rc.setSpacing(10)
        self.lbl_proc_meta = QLabel()
        self.lbl_proc_meta.setObjectName("panelMeta")
        self.lbl_proc_meta.setTextFormat(Qt.RichText)
        self.lbl_proc_meta.setWordWrap(True)
        self.lbl_proc_meta.setMinimumHeight(72)
        self.right_label = QLabel(
            "<div align='center'><span style='font-size:15px;font-weight:600;color:#9aa3b8'>İşlenmiş önizleme</span><br/>"
            "<span style='font-size:12px;color:#5f677a'>İşlemler bu panelde</span></div>"
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
        self.tabs.setMinimumHeight(340)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(top_inner)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([470, 350])
        self.main_splitter.setChildrenCollapsible(False)
        ml.addWidget(self.main_splitter, 1)

        self.setup_tabs()
        self.setup_toolbar()
        self._apply_responsive_ui(force=True)
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

        ar = QAction("Sıfırla (Reset)", self)
        ar.triggered.connect(self.reset_image)
        tb.addAction(ar)

        ab = QAction("Toplu İşlem (Batch)", self)
        ab.triggered.connect(self.batch_process)
        tb.addAction(ab)

    def _tab_page_layout(self):
        page = QWidget()
        page.setObjectName("tabPage")
        page.setAttribute(Qt.WA_StyledBackground, True)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("tabScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setObjectName("tabContent")
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setHorizontalSpacing(16)
        content_layout.setVerticalSpacing(16)
        content_layout.setColumnStretch(0, 1)
        content_layout.setColumnStretch(1, 1)

        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        return page, content_layout

    def _card_label(self, text, object_name):
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        return label

    def _value_badge(self, text):
        label = QLabel(text)
        label.setObjectName("valueBadge")
        label.setAlignment(Qt.AlignCenter)
        return label

    def _primary_button(self, text, callback):
        button = QPushButton(text)
        button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        return button

    def _build_control_card(self, title, description, body_layout, action_button=None):
        card = QFrame()
        card.setObjectName("controlCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(self._card_label(title, "cardTitle"))
        layout.addWidget(self._card_label(description, "cardDescription"))
        layout.addLayout(body_layout)
        layout.addStretch(1)
        if action_button is not None:
            layout.addWidget(action_button)
        return card

    def setup_tabs(self):
        self.tabs.clear()

        t1, grid1 = self._tab_page_layout()
        card_gray = self._build_control_card(
            "Gri Dönüşüm",
            "Renkli görüntüyü tek kanallı gri tona çevirir ve sonraki işlemler için sade bir başlangıç sunar.",
            QVBoxLayout(),
            self._primary_button(
                "Gri Dönüşüm Uygula",
                lambda: self.apply_operation("Gri Dönüşüm", preprocessing.rgb_to_gray),
            ),
        )
        self.slider_binary = QSlider(Qt.Horizontal)
        self.slider_binary.setRange(0, 255)
        self.slider_binary.setValue(128)
        self.label_binary_val = self._value_badge("128")
        self.slider_binary.valueChanged.connect(lambda v: self.label_binary_val.setText(str(v)))
        v2 = QVBoxLayout()
        v2.setSpacing(10)
        v2.addWidget(self._card_label("Eşik Değeri", "fieldLabel"))
        row_binary = QHBoxLayout()
        row_binary.setSpacing(10)
        row_binary.addWidget(self.slider_binary, 1)
        row_binary.addWidget(self.label_binary_val)
        v2.addLayout(row_binary)
        card_binary = self._build_control_card(
            "Binary Dönüşüm",
            "Parlaklık eşiğine göre görüntüyü siyah ve beyaz katmanlara ayırır.",
            v2,
            self._primary_button(
                "Binary Dönüşüm Uygula",
                lambda: self.apply_operation(
                    f"Binary (T:{self.slider_binary.value()})",
                    preprocessing.gray_to_binary,
                    self.slider_binary.value(),
                ),
            ),
        )
        card_hsv = self._build_control_card(
            "HSV Dönüşüm",
            "Renk tonu, doygunluk ve parlaklığı ayrı katmanlarda incelemek için uygundur.",
            QVBoxLayout(),
            self._primary_button(
                "HSV Dönüşüm Uygula",
                lambda: self.apply_operation("HSV", preprocessing.rgb_to_hsv),
            ),
        )
        grid1.addWidget(card_gray, 0, 0)
        grid1.addWidget(card_binary, 0, 1)
        grid1.addWidget(card_hsv, 1, 0, 1, 2)
        self.tabs.addTab(t1, "Ön İşleme")

        t2, grid2 = self._tab_page_layout()
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(-180, 180)
        self.spin_angle.setValue(45)
        rotate_body = QVBoxLayout()
        rotate_body.setSpacing(10)
        rotate_body.addWidget(self._card_label("Açı", "fieldLabel"))
        rotate_body.addWidget(self.spin_angle)
        card_rotate = self._build_control_card(
            "Döndürme",
            "Belgeyi ya da fotoğrafı doğru eksene hizalamak için kontrollü döndürme uygular.",
            rotate_body,
            self._primary_button(
                "Döndür",
                lambda: self.apply_operation(
                    f"Döndürme ({self.spin_angle.value()}°)",
                    geometry.rotate_image,
                    float(self.spin_angle.value()),
                ),
            ),
        )

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
        crop_body = QGridLayout()
        crop_body.setHorizontalSpacing(12)
        crop_body.setVerticalSpacing(10)
        crop_body.addWidget(self._card_label("X", "fieldLabel"), 0, 0)
        crop_body.addWidget(self.spin_x, 0, 1)
        crop_body.addWidget(self._card_label("Y", "fieldLabel"), 0, 2)
        crop_body.addWidget(self.spin_y, 0, 3)
        crop_body.addWidget(self._card_label("Genişlik", "fieldLabel"), 1, 0)
        crop_body.addWidget(self.spin_w, 1, 1)
        crop_body.addWidget(self._card_label("Yükseklik", "fieldLabel"), 1, 2)
        crop_body.addWidget(self.spin_h, 1, 3)
        card_crop = self._build_control_card(
            "Kırpma",
            "Sadece ihtiyaç duyulan bölgeye odaklanmak için kırpma alanını rahatça ayarla.",
            crop_body,
            self._primary_button(
                "Kırp",
                lambda: self.apply_operation(
                    "Kırpma",
                    geometry.crop_image,
                    self.spin_x.value(),
                    self.spin_y.value(),
                    self.spin_w.value(),
                    self.spin_h.value(),
                ),
            ),
        )

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(1, 40)
        self.slider_zoom.setValue(10)
        self.label_zoom = self._value_badge("1.0x")
        self.slider_zoom.valueChanged.connect(lambda v: self.label_zoom.setText(f"{v/10.0}x"))
        zoom_body = QVBoxLayout()
        zoom_body.setSpacing(10)
        zoom_body.addWidget(self._card_label("Yakınlaştırma Oranı", "fieldLabel"))
        row_zoom = QHBoxLayout()
        row_zoom.setSpacing(10)
        row_zoom.addWidget(self.slider_zoom, 1)
        row_zoom.addWidget(self.label_zoom)
        zoom_body.addLayout(row_zoom)
        card_zoom = self._build_control_card(
            "Zoom",
            "Detayları daha yakından incelemek veya görüntü ölçeğini değiştirmek için kullan.",
            zoom_body,
            self._primary_button(
                "Zoom Uygula",
                lambda: self.apply_operation(
                    f"Zoom ({self.slider_zoom.value()/10.0}x)",
                    geometry.zoom_image,
                    self.slider_zoom.value() / 10.0,
                ),
            ),
        )
        grid2.addWidget(card_rotate, 0, 0)
        grid2.addWidget(card_crop, 0, 1)
        grid2.addWidget(card_zoom, 1, 0, 1, 2)
        self.tabs.addTab(t2, "Geometrik Düzeltme")

        t3, grid3 = self._tab_page_layout()
        hist_body = QVBoxLayout()
        hist_body.setSpacing(10)
        hist_actions = QHBoxLayout()
        hist_actions.setSpacing(10)
        hist_actions.addWidget(self._primary_button("Histogram Analizini Göster", self.show_histogram_dialog))
        hist_actions.addWidget(self._primary_button("Histogram Germe", self._apply_histogram_stretch))
        hist_body.addLayout(hist_actions)
        card_hist = self._build_control_card(
            "Histogram Araçları",
            "Parlaklık dağılımını inceleyip ton aralığını daha dengeli hale getirmek için kullan.",
            hist_body,
        )

        self.slider_cont = QSlider(Qt.Horizontal)
        self.slider_cont.setRange(1, 30)
        self.slider_cont.setValue(10)
        self.label_cont = self._value_badge("1.0x")
        self.slider_cont.valueChanged.connect(lambda v: self.label_cont.setText(f"{v/10.0}x"))
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(1, 50)
        self.slider_gamma.setValue(10)
        self.label_gamma = self._value_badge("γ=1.0")
        self.slider_gamma.valueChanged.connect(lambda v: self.label_gamma.setText(f"γ={v/10.0:.1f}"))
        tone_body = QVBoxLayout()
        tone_body.setSpacing(12)
        tone_body.addWidget(self._card_label("Kontrast", "fieldLabel"))
        row_cont = QHBoxLayout()
        row_cont.setSpacing(10)
        row_cont.addWidget(self.slider_cont, 1)
        row_cont.addWidget(self.label_cont)
        tone_body.addLayout(row_cont)
        tone_body.addWidget(self._card_label("Gamma", "fieldLabel"))
        row_gamma = QHBoxLayout()
        row_gamma.setSpacing(10)
        row_gamma.addWidget(self.slider_gamma, 1)
        row_gamma.addWidget(self.label_gamma)
        tone_body.addLayout(row_gamma)
        tone_actions = QHBoxLayout()
        tone_actions.setSpacing(10)
        tone_actions.addWidget(
            self._primary_button(
                "Kontrast Artır",
                lambda: self.apply_operation(
                    f"Kontrast ({self.slider_cont.value()/10.0}x)",
                    enhancement.contrast_enhancement,
                    self.slider_cont.value() / 10.0,
                ),
            )
        )
        tone_actions.addWidget(
            self._primary_button(
                "Gamma Uygula",
                lambda: self.apply_operation(
                    f"Gamma ({self.slider_gamma.value()/10.0:.1f})",
                    enhancement.gamma_correction,
                    self.slider_gamma.value() / 10.0,
                ),
            )
        )
        tone_body.addLayout(tone_actions)
        card_tone = self._build_control_card(
            "Kontrast ve Gamma",
            "Soluk ya da düşük kontrastlı içerikleri daha okunur hale getirmek için ton ayarı yap.",
            tone_body,
        )

        self.slider_sp = QSlider(Qt.Horizontal)
        self.slider_sp.setRange(1, 100)
        self.slider_sp.setValue(5)
        self.label_sp = self._value_badge("%5")
        self.slider_sp.valueChanged.connect(lambda v: self.label_sp.setText(f"%{v}"))
        noise_body = QVBoxLayout()
        noise_body.setSpacing(10)
        noise_body.addWidget(self._card_label("Gürültü Oranı", "fieldLabel"))
        row_noise = QHBoxLayout()
        row_noise.setSpacing(10)
        row_noise.addWidget(self.slider_sp, 1)
        row_noise.addWidget(self.label_sp)
        noise_body.addLayout(row_noise)
        card_noise = self._build_control_card(
            "Tuz ve Biber Gürültüsü",
            "Filtre etkilerini test etmek veya örnek bozulmalar üretmek için kontrollü gürültü ekler.",
            noise_body,
            self._primary_button(
                "Gürültü Ekle",
                lambda: self.apply_operation(
                    f"Tuz/Biber ({self.slider_sp.value()}%)",
                    enhancement.add_salt_pepper_noise,
                    float(self.slider_sp.value()),
                ),
            ),
        )

        pair_body = QVBoxLayout()
        pair_body.setSpacing(10)
        pair_top = QHBoxLayout()
        pair_top.setSpacing(10)
        pair_top.addWidget(self._primary_button("Ortalama (2. dosya)", self.do_image_averaging))
        pair_top.addWidget(self._primary_button("Ekleme", self.do_image_addition))
        pair_body.addLayout(pair_top)
        pair_body.addWidget(self._primary_button("Bölme", self.do_image_division))
        card_pair = self._build_control_card(
            "İki Görüntü İşlemleri",
            "İkinci bir dosya ile karşılaştırma, ekleme ve oranlama işlemlerini daha rahat yap.",
            pair_body,
        )
        grid3.addWidget(card_hist, 0, 0)
        grid3.addWidget(card_tone, 0, 1)
        grid3.addWidget(card_noise, 1, 0)
        grid3.addWidget(card_pair, 1, 1)
        self.tabs.addTab(t3, "İyileştirme")

        t4, grid4 = self._tab_page_layout()
        self.spin_mean_k = QSpinBox()
        self.spin_mean_k.setRange(3, 15)
        self.spin_mean_k.setSingleStep(2)
        self.spin_mean_k.setValue(3)
        mean_body = QVBoxLayout()
        mean_body.setSpacing(10)
        mean_body.addWidget(self._card_label("Kernel Boyutu", "fieldLabel"))
        mean_body.addWidget(self.spin_mean_k)
        card_mean = self._build_control_card(
            "Mean Filtresi",
            "Genel gürültüyü azaltarak geçişleri daha yumuşak ve dengeli hale getirir.",
            mean_body,
            self._primary_button(
                "Mean Uygula",
                lambda: self.apply_operation("Mean", filtering.mean_filter, self.spin_mean_k.value()),
            ),
        )

        self.combo_median_main = QComboBox()
        self.combo_median_main.addItems(["3", "5", "7", "9"])
        median_body = QVBoxLayout()
        median_body.setSpacing(10)
        median_body.addWidget(self._card_label("Kernel Boyutu", "fieldLabel"))
        median_body.addWidget(self.combo_median_main)
        card_median = self._build_control_card(
            "Median Filtresi",
            "Özellikle tuz-biber gürültüsünü azaltırken kenar detaylarını daha iyi korur.",
            median_body,
            self._primary_button(
                "Median Uygula",
                lambda: self.apply_operation(
                    "Median",
                    filtering.median_filter,
                    int(self.combo_median_main.currentText()),
                ),
            ),
        )

        self.spin_gauss_sigma = QDoubleSpinBox()
        self.spin_gauss_sigma.setRange(0.5, 4.0)
        self.spin_gauss_sigma.setValue(1.0)
        gaussian_body = QVBoxLayout()
        gaussian_body.setSpacing(10)
        gaussian_body.addWidget(self._card_label("Sigma Değeri", "fieldLabel"))
        gaussian_body.addWidget(self.spin_gauss_sigma)
        card_gaussian = self._build_control_card(
            "Gaussian Blur",
            "Doğal bir yumuşatma uygular ve sert ton geçişlerini daha dengeli hale getirir.",
            gaussian_body,
            self._primary_button(
                "Gaussian Uygula",
                lambda: self.apply_operation("Gaussian", gaussian_blur_sigma, self.spin_gauss_sigma.value()),
            ),
        )

        self.spin_unsharp_main = QDoubleSpinBox()
        self.spin_unsharp_main.setRange(0.1, 5.0)
        self.spin_unsharp_main.setValue(1.5)
        unsharp_body = QVBoxLayout()
        unsharp_body.setSpacing(10)
        unsharp_body.addWidget(self._card_label("Keskinleştirme Miktarı", "fieldLabel"))
        unsharp_body.addWidget(self.spin_unsharp_main)
        card_unsharp = self._build_control_card(
            "Unsharp Mask",
            "Belgedeki detayları belirginleştirir ve hafif bulanıklıkları toparlar.",
            unsharp_body,
            self._primary_button(
                "Unsharp Uygula",
                lambda: self.apply_operation("Unsharp", filtering.unsharp_mask, self.spin_unsharp_main.value()),
            ),
        )
        grid4.addWidget(card_mean, 0, 0)
        grid4.addWidget(card_median, 0, 1)
        grid4.addWidget(card_gaussian, 1, 0)
        grid4.addWidget(card_unsharp, 1, 1)
        self.tabs.addTab(t4, "Filtreleme")

        t5, grid5 = self._tab_page_layout()
        card_edge = self._build_control_card(
            "Prewitt Kenar Analizi",
            "Belgedeki ana hatları ve şekil sınırlarını net biçimde ortaya çıkarır.",
            QVBoxLayout(),
            self._primary_button(
                "Prewitt Uygula",
                lambda: self.apply_operation("Prewitt", analysis.prewitt_edge_detection),
            ),
        )
        self.combo_morph_shape = QComboBox()
        self.combo_morph_shape.addItems(["rect", "cross", "ellipse"])
        self.combo_morph = QComboBox()
        for s in (3, 5, 7, 9):
            self.combo_morph.addItem(str(s))
        morph_body = QVBoxLayout()
        morph_body.setSpacing(12)
        hm = QGridLayout()
        hm.setHorizontalSpacing(12)
        hm.setVerticalSpacing(10)
        hm.addWidget(self._card_label("Şekil", "fieldLabel"), 0, 0)
        hm.addWidget(self.combo_morph_shape, 0, 1)
        hm.addWidget(self._card_label("Boyut", "fieldLabel"), 0, 2)
        hm.addWidget(self.combo_morph, 0, 3)
        morph_body.addLayout(hm)
        h2 = QGridLayout()
        h2.setHorizontalSpacing(10)
        h2.setVerticalSpacing(10)
        button_index = 0
        for txt, fn in (
            ("Erozyon", analysis.erosion),
            ("Dilation", analysis.dilation),
            ("Opening", analysis.opening),
            ("Closing", analysis.closing),
        ):
            b = self._primary_button(
                txt,
                lambda checked=False, f=fn, name=txt: self.apply_operation(
                    name, f, self._morph_structuring_element(), "auto"
                ),
            )
            h2.addWidget(b, button_index // 2, button_index % 2)
            button_index += 1
        morph_body.addLayout(h2)
        card_morph = self._build_control_card(
            "Morfoloji Araçları",
            "Küçük bozulmaları temizlemek ve yapısal formları güçlendirmek için temel morfolojik işlemleri sunar.",
            morph_body,
        )
        grid5.addWidget(card_edge, 0, 0)
        grid5.addWidget(card_morph, 0, 1)
        self.tabs.addTab(t5, "Analiz & Morfoloji")

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
                return "-"
            if len(img.shape) == 3:
                return f"{img.shape[2]} kanal (BGR)"
            return "Tek kanal"

        def card(title, lines):
            esc = "<br/>".join(html.escape(x) for x in lines)
            return (
                f"<div><span style='font-weight:600'>{html.escape(title)}</span>"
                f"<span style='display:block;margin-top:6px;color:#9ca3b8;font-size:{self._scale_px(12)}px'>{esc}</span></div>"
            )

        if self.original_image is None:
            self.lbl_orig_meta.setText(
                card("Kaynak", ["Dosya: -", "Çözünürlük: -", "Kanallar: -"])
            )
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "-"
            ho, wo = self.original_image.shape[:2]
            self.lbl_orig_meta.setText(
                card("Kaynak", [f"Dosya: {fn}", f"Çözünürlük: {wo}x{ho}", f"Kanal: {ch(self.original_image)}"])
            )
        if self.current_image is None:
            self.lbl_proc_meta.setText(card("İşlenmiş", ["-", "-", "-"]))
        else:
            fn = os.path.basename(self.image_path) if self.image_path else "-"
            hc, wc = self.current_image.shape[:2]
            last = self.history[-1][0] if self.history else "-"
            self.lbl_proc_meta.setText(
                card(
                    "İşlenmiş",
                    [f"Dosya: {fn}", f"Çözünürlük: {wc}x{hc}", f"Kanal: {ch(self.current_image)}", f"Son: {last}"],
                )
            )

    def _morph_structuring_element(self):
        return analysis.get_structuring_element(
            self.combo_morph_shape.currentText(),
            int(self.combo_morph.currentText()),
        )

    def _apply_histogram_stretch(self):
        self.apply_operation("Histogram Germe", enhancement.histogram_stretching)

    def _should_fit_display(self, img_array):
        if img_array is None:
            return True
        if not self.history:
            return True
        return not self.history[-1][0].startswith("Zoom ")

    def apply_operation(self, op_name, module_func, *args):
        if self.current_image is None:
            QMessageBox.warning(self, "Hata", "Önce görüntü yükleyin!")
            return
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
        self.apply_operation("Image Averaging", enhancement.image_averaging, img2)

    def do_image_addition(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "2. görüntü", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            img2 = safe_imread(fn)
            if img2 is not None:
                self.apply_operation("Ekleme", geometry.add_images, img2)

    def do_image_division(self):
        if self.current_image is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "2. görüntü", "", "Görüntü (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            img2 = safe_imread(fn)
            if img2 is not None:
                self.apply_operation("Bölme", geometry.divide_images, img2)

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
        dlg = BatchProcessDialog(self.processing_modules, self)
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
        h, w = img_u8.shape[:2]
        if len(img_u8.shape) == 3:
            rgb = img_u8[:, :, ::-1].copy()
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            g = np.ascontiguousarray(img_u8)
            qimg = QImage(g.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

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
                        self.update_image_display(self.original_image, self.left_label, self.left_scroll)
                    elif label is self.right_label and self.current_image is not None:
                        self.update_image_display(self.current_image, self.right_label, self.right_scroll)
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
                    img_array = cv2.resize(img_array, (rw, rh), interpolation=cv2.INTER_AREA)
        pm_show = self._numpy_to_qpixmap(img_array)
        label.setPixmap(pm_show)
        label.adjustSize()

    def _refit_image_panels(self):
        try:
            if self.original_image is not None:
                self.update_image_display(self.original_image, self.left_label, self.left_scroll)
            if self.current_image is not None:
                self.update_image_display(self.current_image, self.right_label, self.right_scroll)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_ui()
        self._resize_fit_timer.start(120)
