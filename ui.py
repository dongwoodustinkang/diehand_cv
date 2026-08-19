"""Defect Detector의 macOS 스타일 화면과 사용자 상호작용을 정의한다."""

from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from contour import create_detection_visualization
from styles import APP_STYLESHEET


IMAGE_EXTS = {".tif", ".tiff"}
DEFAULT_DIR = "dataset/"


class MainWindow(QMainWindow):
    """A/B TIFF의 B 페이지 컨투어를 확인하는 메인 창."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("불량 검출기")
        self.resize(1280, 760)
        self.setMinimumSize(960, 620)

        self.image_paths = []
        self.current_index = -1
        self.original_pixmap = QPixmap() # 원본 이미지
        self.result_pixmap = QPixmap() # B 페이지 이미지
        self.analysis_preview_pixmap = QPixmap()
        self.source_preview_pixmap = QPixmap()
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addLayout(self._create_header())
        root.addLayout(self._create_workspace(), stretch=1)
        root.addWidget(self._create_info_card())
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def _create_header(self):
        header = QHBoxLayout()
        header.setSpacing(12)

        title_group = QVBoxLayout()
        title_group.setSpacing(2)
        title = QLabel("Die 이상 탐지 프로그램")
        title.setObjectName("windowTitle")
        subtitle = QLabel("B 페이지 컨투어 첫 접점 분석")
        subtitle.setObjectName("subtitle")
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        header.addLayout(title_group)
        header.addStretch()

        self.prev_btn = QPushButton("‹")
        self.prev_btn.setObjectName("navigationButton")
        self.prev_btn.setToolTip("이전 이미지 (←)")
        self.next_btn = QPushButton("›")
        self.next_btn.setObjectName("navigationButton")
        self.next_btn.setToolTip("다음 이미지 (→)")
        self.index_label = QLabel("0 / 0")
        self.index_label.setObjectName("indexLabel")
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)
        header.addWidget(self.prev_btn)
        header.addWidget(self.index_label)
        header.addWidget(self.next_btn)

        return header

    def _create_workspace(self):
        """A 페이지, B 페이지, 컨투어 분석을 같은 너비의 3열로 배치한다."""
        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        source_card, self.image_label = self._create_image_card(
            "A 페이지", "A/B TIFF 이미지를 불러오세요."
        )
        result_card, self.result_label = self._create_image_card(
            "B 페이지", "B 페이지 컨투어 분석 결과가 표시됩니다."
        )
        workspace.addWidget(source_card, stretch=1)
        workspace.addWidget(result_card, stretch=1)
        workspace.addWidget(self._create_control_sidebar(), stretch=1)
        return workspace

    def _create_image_card(self, title_text, empty_text):
        card = QFrame()
        card.setObjectName("imageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        label = self._create_image_label(empty_text)
        layout.addWidget(label, stretch=1)
        return card, label

    def _create_info_card(self):
        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_card.setMinimumHeight(106)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 13, 16, 13)
        info_layout.setSpacing(6)
        info_title = QLabel("검사 상세 정보")
        info_title.setObjectName("sectionTitle")
        self.info_label = QLabel("이미지를 불러오면 상세 정보가 표시됩니다.")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setTextFormat(Qt.PlainText)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_label)
        return info_card

    def _create_control_sidebar(self):
        controls = QFrame()
        controls.setObjectName("controlCard")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setSpacing(10)

        controls_title = QLabel("컨투어 분석")
        controls_title.setObjectName("sectionTitle")
        controls_layout.addWidget(controls_title)
        self.analysis_preview_label = self._create_preview_card(
            controls_layout,
            "B 페이지 컨투어",
            "근사 컨투어 영역의 B 페이지 원본이 표시됩니다.",
        )
        self.source_preview_label = self._create_preview_card(
            controls_layout,
            "A 페이지 동일 컨투어",
            "같은 컨투어 좌표의 A 페이지 원본이 표시됩니다.",
        )

        self.import_btn = QPushButton("불러오기")
        self.import_btn.setObjectName("secondaryButton")
        self.detect_btn = QPushButton("검출하기")
        self.detect_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self.on_import)
        self.detect_btn.clicked.connect(self.on_detect)
        controls_layout.addWidget(self.import_btn)
        controls_layout.addWidget(self.detect_btn)

        return controls

    def _create_preview_card(self, parent_layout, title_text, empty_text):
        preview_card = QFrame()
        preview_card.setObjectName("imageCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(6)

        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        preview_layout.addWidget(title)

        preview_label = QLabel(empty_text)
        preview_label.setObjectName("imagePreview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_label.setMinimumHeight(120)
        preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(preview_label)
        parent_layout.addWidget(preview_card, stretch=1)
        return preview_label

    def _create_image_label(self, message):
        label = QLabel(message)
        label.setObjectName("imagePreview")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setMinimumHeight(360)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return label

    # ---------------- Import ----------------
    def on_import(self):
        dialog = QFileDialog(self, "이미지 불러오기", DEFAULT_DIR)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("TIFF 이미지 (*.tif *.tiff)")
        dialog.setLabelText(QFileDialog.Accept, "선택")

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected = [Path(path) for path in dialog.selectedFiles()]
        if not selected:
            return

        if len(selected) == 1 and selected[0].is_dir():
            paths = sorted(
                path for path in selected[0].iterdir()
                if path.suffix.lower() in IMAGE_EXTS
            )
            if not paths:
                QMessageBox.warning(self, "불러오기", "폴더에 이미지가 없습니다.")
                return
            self._set_image_list(paths)
            return

        files = [path for path in selected if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
        if files:
            self._set_image_list(files)

    def _set_image_list(self, paths):
        self.image_paths = paths
        self.current_index = 0
        self._show_current()

    # ---------------- Navigation ----------------
    def show_prev(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def show_next(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._show_current()

    def _show_current(self):
        path = self.image_paths[self.current_index]
        self.original_pixmap = QPixmap(str(path))
        self._set_scaled_pixmap(self.image_label, self.original_pixmap)
        self.index_label.setText(f"{self.current_index + 1} / {len(self.image_paths)}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)

        self._detect_current_image()

    def on_detect(self):
        if not self.image_paths:
            QMessageBox.information(self, "검출", "먼저 이미지를 불러오세요.")
            return
        self._detect_current_image()

    def _detect_current_image(self):
        """한 장의 검출 시간을 재서 결과 정보에 이미지/초 처리 속도를 표시한다."""
        path = self.image_paths[self.current_index]
        started_at = perf_counter()
        try:
            result_image, result = create_detection_visualization(path)
        except ValueError as error:
            self._clear_result("검출에 실패했습니다.")
            QMessageBox.critical(self, "검출", str(error))
            return

        elapsed_seconds = perf_counter() - started_at
        images_per_second = 1 / max(elapsed_seconds, 0.000001)
        self._show_result_image(result_image)
        self._show_analysis_preview(result.analysis_preview)
        self._show_source_preview(result.source_preview)
        self._update_info_label(path, result, elapsed_seconds, images_per_second)

    def _show_result_image(self, bgr_image):
        self.result_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(self.result_label, self.result_pixmap)

    def _show_analysis_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.analysis_preview_pixmap = QPixmap()
            self.analysis_preview_label.setPixmap(QPixmap())
            self.analysis_preview_label.setText("검출된 분석 개체가 없습니다.")
            return

        self.analysis_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(self.analysis_preview_label, self.analysis_preview_pixmap)

    def _show_source_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.source_preview_pixmap = QPixmap()
            self.source_preview_label.setPixmap(QPixmap())
            self.source_preview_label.setText("표시할 A 페이지 좌표가 없습니다.")
            return

        self.source_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(self.source_preview_label, self.source_preview_pixmap)

    @staticmethod
    def _pixmap_from_image(image):
        image = image.copy()
        height, width, channels = image.shape
        if channels == 4:
            rgba_image = image[:, :, [2, 1, 0, 3]].copy()
            qimage = QImage(
                rgba_image.tobytes(), width, height, width * 4, QImage.Format_RGBA8888
            )
        else:
            qimage = QImage(
                image.tobytes(), width, height, width * 3, QImage.Format_BGR888
            )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _set_scaled_pixmap(label, pixmap, preview_scale=1.0):
        target_size = label.size()
        if preview_scale != 1.0:
            target_size.setWidth(round(target_size.width() * preview_scale))
            target_size.setHeight(round(target_size.height() * preview_scale))
        label.setPixmap(
            pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _clear_result(self, message):
        self.result_pixmap = QPixmap()
        self.result_label.setPixmap(QPixmap())
        self.result_label.setText(message)
        self.analysis_preview_pixmap = QPixmap()
        self.analysis_preview_label.setPixmap(QPixmap())
        self.analysis_preview_label.setText("근사 컨투어 영역의 B 페이지 원본이 표시됩니다.")
        self.source_preview_pixmap = QPixmap()
        self.source_preview_label.setPixmap(QPixmap())
        self.source_preview_label.setText("같은 컨투어 좌표의 A 페이지 원본이 표시됩니다.")

    def _update_info_label(
        self, path, result=None, elapsed_seconds=None, images_per_second=None
    ):
        lines = [
            f"1. 파일 위치 :   {path.name}",
            f"2. 이미지 크기  {self.original_pixmap.width()} × {self.original_pixmap.height()} px",
        ]
        if result is not None:
            lines.append(f"3. B 페이지 외곽 컨투어 개수 : {len(result.contours)}개")
            point_count = sum(
                len(measurement.top_points)
                + len(measurement.bottom_points)
                + len(measurement.left_points)
                + len(measurement.right_points)
                for measurement in result.measurements
            )
            lines.append(f"4. 4면 첫 접점 개수 : {point_count}개")
            if elapsed_seconds is not None and images_per_second is not None:
                lines.append(
                    f"5. 처리 시간  {elapsed_seconds * 1000:.1f} ms · "
                    f"{images_per_second:.2f} image/sec"
                )
        self.info_label.setText("\n".join(lines))

    def _refresh_scaled_pixmaps(self):
        if not self.original_pixmap.isNull():
            self._set_scaled_pixmap(self.image_label, self.original_pixmap)
        if not self.result_pixmap.isNull():
            self._set_scaled_pixmap(self.result_label, self.result_pixmap)
        if not self.analysis_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.analysis_preview_label, self.analysis_preview_pixmap
            )
        if not self.source_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.source_preview_label, self.source_preview_pixmap
            )
    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_scaled_pixmaps()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled_pixmaps()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.show_prev()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)
