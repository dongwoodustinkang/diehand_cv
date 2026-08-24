"""Defect Detector의 macOS 스타일 화면과 사용자 상호작용을 정의한다."""
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from contour import (
    create_detection_visualization,
    get_primary_contact_reference_point,
)
from styles import APP_STYLESHEET


IMAGE_EXTS = {".tif", ".tiff"}
DEFAULT_DIR = "dataset/"
PREVIEW_SCALE = 0.8
DEV_IMAGE_DIR = Path("/Users/dongwookang/diehand_cv/dataset/side/total")
# DEV_IMAGE_DIR = Path("/Users/dongwookang/diehand")
CAPTURE_ROOT = Path(__file__).resolve().parent / "captures"

class ClickableImageLabel(QLabel):
    """클릭 이벤트를 전달하는 이미지 라벨."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class ImageModal(QDialog):
    """이미지를 중앙에서 크게 확인하고 다시 클릭해 닫는 모달."""

    def __init__(self, parent):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self.setObjectName("imageModal")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("imageModalCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        card_layout.setSpacing(8)
        self.title_label = QLabel()
        self.title_label.setObjectName("imageModalTitle")
        self.image_scroll = QScrollArea()
        self.image_scroll.setObjectName("imageModalScroll")
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_label = ClickableImageLabel()
        self.image_label.setObjectName("imageModalPreview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_scroll.setWidget(self.image_label)

        zoom_controls = QHBoxLayout()
        zoom_controls.setSpacing(6)
        zoom_controls.addStretch()
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoomButton")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_reset_button = QPushButton("맞춤")
        self.zoom_reset_button.setObjectName("zoomResetButton")
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoomButton")
        zoom_controls.addWidget(self.zoom_out_button)
        zoom_controls.addWidget(self.zoom_label)
        zoom_controls.addWidget(self.zoom_in_button)
        zoom_controls.addWidget(self.zoom_reset_button)
        zoom_controls.addStretch()

        self.hint_label = QLabel("이미지를 다시 클릭하면 닫힙니다.")
        self.hint_label.setObjectName("imageModalHint")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.image_label.clicked.connect(self.accept)
        self.zoom_out_button.clicked.connect(lambda: self._change_zoom(1 / 1.25))
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_in_button.clicked.connect(lambda: self._change_zoom(1.25))
        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.image_scroll, stretch=1)
        card_layout.addLayout(zoom_controls)
        card_layout.addWidget(self.hint_label)
        layout.addWidget(card, stretch=1)

    def show_pixmap(self, pixmap, title):
        if pixmap.isNull():
            return

        parent_size = self.parentWidget().size()
        self.resize(
            max(420, min(round(parent_size.width() * 0.82), 1500)),
            max(360, min(round(parent_size.height() * 0.82), 1000)),
        )
        parent_center = self.parentWidget().mapToGlobal(
            self.parentWidget().rect().center()
        )
        self.move(parent_center - self.rect().center())
        self._pixmap = pixmap
        self.zoom_factor = 1.0
        self.title_label.setText(title)
        self._refresh_pixmap()
        self.exec_()

    def _refresh_pixmap(self):
        if self._pixmap.isNull():
            return
        viewport_size = self.image_scroll.viewport().size()
        if viewport_size.width() <= 1 or viewport_size.height() <= 1:
            return
        fitted = self._pixmap.scaled(
            viewport_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        scaled = self._pixmap.scaled(
            max(1, round(fitted.width() * self.zoom_factor)),
            max(1, round(fitted.height() * self.zoom_factor)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedSize(scaled.size())
        self.zoom_label.setText(f"{round(self.zoom_factor * 100)}%")

    def _change_zoom(self, factor):
        self.zoom_factor = min(5.0, max(0.25, self.zoom_factor * factor))
        self._refresh_pixmap()

    def _reset_zoom(self):
        self.zoom_factor = 1.0
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_pixmap()


class MainWindow(QMainWindow):
    """A/B TIFF의 B 페이지 컨투어를 확인하는 메인 창."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("불량 검출기")
        self.resize(2120, 880)
        self.setMinimumSize(1200, 720)

        self.image_paths = []
        self.current_index = -1
        self.max_ball_count = 3
        self.original_pixmap = QPixmap() # 원본 이미지
        self.result_pixmap = QPixmap() # B 페이지 이미지
        self.analysis_preview_pixmap = QPixmap()
        self.source_preview_pixmap = QPixmap()
        self.capture_session_dir = None
        self.capture_btn = QPushButton()
        self.capture_btn.setObjectName("captureIconButton")
        self.capture_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.capture_btn.setToolTip("현재 화면 캡처 저장")
        self.capture_btn.clicked.connect(self.on_capture)
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
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def _create_header(self):
        header = QHBoxLayout()
        header.setSpacing(12)

        title_group = QVBoxLayout()
        title_group.setSpacing(2)
        title = QLabel("이상탐지 프로그램")
        title.setObjectName("windowTitle")
        self.header_metadata_label = QLabel(
            "파일을 불러오면 처리 정보가 표시됩니다."
        )
        self.header_metadata_label.setObjectName("headerMetadata")
        title_group.addWidget(title)
        title_group.addWidget(self.header_metadata_label)
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
        header.addWidget(self.capture_btn)

        return header

    def _create_workspace(self):
        """원본·분석·설정을 역할에 맞는 세 영역으로 배치한다."""
        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        source_column = QVBoxLayout()
        source_column.setSpacing(16)
        source_column.addWidget(self._create_source_card(), stretch=4)
        source_column.addWidget(self._create_log_card(), stretch=1)

        workspace.addLayout(source_column, stretch=5)
        workspace.addWidget(self._create_analysis_card(), stretch=3)
        workspace.addWidget(self._create_control_sidebar(), stretch=2)
        return workspace

    def _create_source_card(self):
        card = QFrame()
        card.setObjectName("imageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("원본 이미지")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        layout.addWidget(self._create_divider())

        image_pair = QHBoxLayout()
        image_pair.setSpacing(14)

        source_panel, self.image_label = self._create_image_panel(
            "A 페이지", "A/B TIFF 이미지를 불러오세요."
        )
        result_panel, self.result_label = self._create_image_panel(
            "B 페이지 결과", "B 페이지 컨투어 분석 결과가 표시됩니다."
        )
        self.image_label.clicked.connect(
            lambda: self._show_image_modal(self.original_pixmap, "A 페이지")
        )
        self.result_label.clicked.connect(
            lambda: self._show_image_modal(self.result_pixmap, "B 페이지 결과")
        )
        image_pair.addWidget(source_panel, stretch=1)
        image_pair.addWidget(result_panel, stretch=1)
        layout.addLayout(image_pair, stretch=1)
        return card

    def _create_image_panel(self, title_text, empty_text):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("imagePaneTitle")
        layout.addWidget(title)

        label = self._create_image_label(empty_text)
        layout.addWidget(label, stretch=1)
        return panel, label

    def _create_log_card(self):
        info_card = QFrame()
        info_card.setObjectName("logCard")
        info_card.setMinimumHeight(126)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 13, 16, 13)
        info_layout.setSpacing(6)
        info_title = QLabel("프로그램 로그")
        info_title.setObjectName("cardTitle")
        self.info_label = QLabel("이미지를 불러오면 상세 정보가 표시됩니다.")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setTextFormat(Qt.PlainText)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_label)
        return info_card

    def _create_analysis_card(self):
        card = QFrame()
        card.setObjectName("analysisCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("컨투어 이미지")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        layout.addWidget(self._create_divider())

        self.analysis_preview_label = self._create_preview_section(
            layout,
            "B 페이지 컨투어",
            "검출된 컨투어 영역의 B 페이지 원본이 표시됩니다.",
        )
        self.source_preview_label = self._create_preview_section(
            layout,
            "A 페이지 기준선 교점 사각형",
            "네 기준선 교점 형태를 유지한 A 페이지 영역이 표시됩니다.",
        )
        self.analysis_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.analysis_preview_pixmap, "B 페이지 컨투어"
            )
        )
        self.source_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.source_preview_pixmap, "A 페이지 기준선 교점 사각형"
            )
        )
        return card

    def _create_control_sidebar(self):
        controls = QFrame()
        controls.setObjectName("controlCard")
        controls.setMinimumWidth(280)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 16, 18, 18)
        controls_layout.setSpacing(12)

        controls_title = QLabel("탐지 설정")
        controls_title.setObjectName("cardTitle")
        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(self._create_divider())
        controls_layout.addWidget(self._create_readonly_setting(
            "탐지 유형", "측면 컨투어"
        ))
        controls_layout.addWidget(self._create_ball_count_control())
        controls_layout.addWidget(self._create_divider())
        controls_layout.addStretch(1)

        status_box = QFrame()
        status_box.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(12, 18, 12, 18)
        status_layout.setSpacing(7)
        self.status_label = QLabel("분석 대기")
        self.status_label.setObjectName("statusValue")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_detail_label = QLabel("이미지를 불러오세요.")
        self.status_detail_label.setObjectName("statusDetail")
        self.status_detail_label.setAlignment(Qt.AlignCenter)
        self.status_detail_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_detail_label)
        controls_layout.addWidget(status_box)
        controls_layout.addStretch(1)

        self.import_btn = QPushButton("불러오기")
        self.import_btn.setObjectName("secondaryButton")
        self.detect_btn = QPushButton("검출하기")
        self.detect_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self.on_import)
        self.detect_btn.clicked.connect(self.on_detect)
        controls_layout.addWidget(self.import_btn)
        controls_layout.addWidget(self.detect_btn)

        return controls

    @staticmethod
    def _create_divider():
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        return divider

    def _create_readonly_setting(self, label_text, value_text):
        setting_row = QFrame()
        setting_row.setObjectName("settingRow")
        layout = QHBoxLayout(setting_row)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("controlLabel")
        value = QLabel(value_text)
        value.setObjectName("settingValue")
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        return setting_row

    def _create_ball_count_control(self):
        """볼 탐지 로직에서 사용할 최대 탐지 개수 선택기를 만든다."""
        setting_row = QFrame()
        setting_row.setObjectName("settingRow")
        layout = QHBoxLayout(setting_row)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(8)

        label = QLabel("최대 볼 탐지 개수")
        label.setObjectName("controlLabel")
        segment = QFrame()
        segment.setObjectName("ballCountSegment")
        segment_layout = QHBoxLayout(segment)
        segment_layout.setContentsMargins(2, 2, 2, 2)
        segment_layout.setSpacing(1)

        self.max_ball_count_group = QButtonGroup(self)
        self.max_ball_count_group.setExclusive(True)
        self.max_ball_count_buttons = {}
        for count in (2, 3):
            button = QPushButton(str(count))
            button.setObjectName("ballCountSegmentButton")
            button.setCheckable(True)
            self.max_ball_count_group.addButton(button, count)
            self.max_ball_count_buttons[count] = button
            segment_layout.addWidget(button)
        self.max_ball_count_buttons[self.max_ball_count].setChecked(True)
        self.max_ball_count_group.buttonClicked[int].connect(
            self._on_max_ball_count_changed
        )

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(segment)
        return setting_row

    def _on_max_ball_count_changed(self, count):
        self.max_ball_count = count

    def _create_preview_section(self, parent_layout, title_text, empty_text):
        preview_section = QWidget()
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        title = QLabel(title_text)
        title.setObjectName("previewTitle")
        preview_layout.addWidget(title)

        preview_label = ClickableImageLabel(empty_text)
        preview_label.setObjectName("imagePreview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_label.setMinimumHeight(160)
        preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(preview_label)
        parent_layout.addWidget(preview_section, stretch=1)
        return preview_label

    def _create_image_label(self, message):
        label = ClickableImageLabel(message)
        label.setObjectName("imagePreview")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setMinimumHeight(300)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return label

    def _show_image_modal(self, pixmap, title):
        if not pixmap.isNull():
            ImageModal(self).show_pixmap(pixmap, title)

    # ---------------- Capture ----------------
    @staticmethod
    def _next_capture_path(capture_dir):
        """기존 캡처 뒤에 이어질 다섯 자리 JPG 파일 경로를 반환한다."""

        sequence_numbers = [
            int(path.stem)
            for path in capture_dir.glob("*.jpg")
            if path.stem.isdecimal() and len(path.stem) == 5
        ]
        next_sequence = max(sequence_numbers, default=0) + 1
        return capture_dir / f"{next_sequence:05d}.jpg"

    def on_capture(self):
        """현재 프로그램 화면을 실행 단위의 날짜·시간 폴더에 JPG로 저장한다."""

        if self.capture_session_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.capture_session_dir = CAPTURE_ROOT / timestamp
            self.capture_session_dir.mkdir(parents=True, exist_ok=True)

        capture_path = self._next_capture_path(self.capture_session_dir)
        if not self.grab().save(str(capture_path), "JPG", quality=95):
            QMessageBox.critical(self, "캡처 저장", "화면 캡처를 저장하지 못했습니다.")
            return

        self.status_detail_label.setText(
            f"캡처 저장 완료: {capture_path.relative_to(CAPTURE_ROOT)}"
        )

    # ---------------- Import ----------------
    def on_import(self):
        # 개발용 임시 로드: 아래 블록을 주석 처리하거나 삭제하면 파일 선택 창만 사용합니다.
        if DEV_IMAGE_DIR.is_dir():
            dev_paths = sorted(
                path
                for path in DEV_IMAGE_DIR.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
            if dev_paths:
                self._set_image_list(dev_paths)
                return

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
        self._set_scaled_pixmap(
            self.analysis_preview_label,
            self.analysis_preview_pixmap,
            preview_scale=PREVIEW_SCALE,
        )

    def _show_source_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.source_preview_pixmap = QPixmap()
            self.source_preview_label.setPixmap(QPixmap())
            self.source_preview_label.setText("표시할 A 페이지 좌표가 없습니다.")
            return

        self.source_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(
            self.source_preview_label,
            self.source_preview_pixmap,
            preview_scale=PREVIEW_SCALE,
        )

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
        self.analysis_preview_label.setText("검출된 컨투어 영역의 B 페이지 원본이 표시됩니다.")
        self.source_preview_pixmap = QPixmap()
        self.source_preview_label.setPixmap(QPixmap())
        self.source_preview_label.setText(
            "네 기준선 교점 형태를 유지한 A 페이지 영역이 표시됩니다."
        )
        self.status_label.setText("분석 실패")
        self.status_label.setStyleSheet("color: #F04452;")
        self.status_detail_label.setText(message)

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
            for index, measurement in enumerate(result.measurements, start=1):
                lines.extend(self._first_contact_log_lines(measurement, index))
            if elapsed_seconds is not None and images_per_second is not None:
                lines.append(
                    f"처리 시간  {elapsed_seconds * 1000:.1f} ms · "
                    f"{images_per_second:.2f} image/sec"
                )
        self.info_label.setText("\n".join(lines))

        if elapsed_seconds is not None and images_per_second is not None:
            self.header_metadata_label.setText(
                f"파일 위치  {path.name} · 처리 시간 {elapsed_seconds * 1000:.1f} ms · "
                f"{images_per_second:.0f} image/sec"
            )
        if result is not None:
            contour_count = len(result.contours)
            self.status_label.setText("분석 완료" if contour_count else "컨투어 미검출")
            self.status_label.setStyleSheet(
                "color: #3182F6;" if contour_count else "color: #F59F00;"
            )
            self.status_detail_label.setText(
                f"B 페이지 외곽 컨투어 {contour_count}개를 확인했습니다."
            )

    @staticmethod
    def _first_contact_log_lines(measurement, contour_index):
        """방향별 첫 접점 중 가장 먼저 닿는 좌표를 로그용 텍스트로 만든다."""

        point_groups = (
            ("상", "y", measurement.top_points, 1, True),
            ("하", "y", measurement.bottom_points, 1, False),
            ("좌", "x", measurement.left_points, 0, True),
            ("우", "x", measurement.right_points, 0, False),
        )

        lines = [f"컨투어 {contour_index} · 가장 먼저 닿는 좌표"]
        for direction, axis, points, coordinate_index, use_minimum in point_groups:
            reference_point = get_primary_contact_reference_point(
                points, coordinate_index, use_minimum
            )
            if reference_point is None:
                lines.append(f"{direction} : 없음")
                continue
            lines.append(
                f"{direction} : {axis}={reference_point[coordinate_index]} "
                f"· 대표점 ({reference_point[0]}, {reference_point[1]})"
            )
        return lines

    def _refresh_scaled_pixmaps(self):
        if not self.original_pixmap.isNull():
            self._set_scaled_pixmap(self.image_label, self.original_pixmap)
        if not self.result_pixmap.isNull():
            self._set_scaled_pixmap(self.result_label, self.result_pixmap)
        if not self.analysis_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.analysis_preview_label,
                self.analysis_preview_pixmap,
                preview_scale=PREVIEW_SCALE,
            )
        if not self.source_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.source_preview_label,
                self.source_preview_pixmap,
                preview_scale=PREVIEW_SCALE,
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
