from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class BatchProcessDialog(QDialog):
    def __init__(self, modules, parent=None):
        super().__init__(parent)
        self.modules = modules
        self.setWindowTitle("Toplu İşlem (Batch) Yöneticisi")
        self.resize(500, 550)
        self.input_files = []
        self.output_folder = ""

        dlg_layout = QVBoxLayout(self)
        grp_io = QGroupBox("Girdi ve çıktı")
        vio = QVBoxLayout()

        h_in = QHBoxLayout()
        btn_in = QPushButton("İşlenecek görüntüleri seç...")
        btn_in.clicked.connect(self.select_in)
        self.lbl_in = QLabel("Henüz dosya seçilmedi")
        h_in.addWidget(btn_in)
        h_in.addWidget(self.lbl_in, 1)
        vio.addLayout(h_in)

        h_out = QHBoxLayout()
        btn_out = QPushButton("Çıktı klasörü seç...")
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
            self,
            "Görüntüleri Seç",
            "",
            "Görüntü (*.png *.jpg *.jpeg *.bmp)",
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
        preprocessing = self.modules["preprocessing"]
        enhancement = self.modules["enhancement"]
        filtering = self.modules["filtering"]
        analysis = self.modules["analysis"]

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
