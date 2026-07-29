import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from contour import create_detection_visualization

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_DIR = "/diehand/Dataset/"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Detector")
        self.resize(1200, 650)

        self.image_paths = []   # 불러온 이미지 경로 리스트
        self.current_index = -1
        self.original_pixmap = QPixmap()
        self.result_pixmap = QPixmap()

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- 좌측 원본 / 우측 검출 결과 표시 영역 ---
        image_row = QHBoxLayout()
        self.image_label = self._create_image_label("Import Image")
        self.result_label = self._create_image_label(
            "컨투어 항목을 선택한 뒤 Detect를 누르세요."
        )

        source_column = QVBoxLayout()
        source_column.addWidget(QLabel("원본 이미지"))
        source_column.addWidget(self.image_label)
        image_row.addLayout(source_column)

        result_column = QVBoxLayout()
        result_column.addWidget(QLabel("검출 결과"))
        result_column.addWidget(self.result_label)
        image_row.addLayout(result_column)
        root.addLayout(image_row)

        # --- 내비게이션 (◀ n/n ▶) ---
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.index_label = QLabel("0 / 0")
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)
        nav.addStretch()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.index_label)
        nav.addWidget(self.next_btn)
        nav.addStretch()
        root.addLayout(nav)

        # --- 하단: 정보 박스 + 버튼 ---
        bottom = QHBoxLayout()

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("border: 1px solid black;")
        self.info_label.setFixedHeight(120)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        bottom.addWidget(self.info_label, stretch=1)

        btn_col = QVBoxLayout()
        self.import_btn = QPushButton("Import..")
        self.detect_btn = QPushButton("Detect")
        self.quit_btn = QPushButton("Quit")
        self.surface_checkbox = QCheckBox("Surface 컨투어")
        self.ball_checkbox = QCheckBox("Ball 컨투어")

        self.import_btn.clicked.connect(self.on_import)
        self.detect_btn.clicked.connect(self.on_detect)
        self.quit_btn.clicked.connect(self.close)
        self.surface_checkbox.toggled.connect(self._on_detection_option_changed)
        self.ball_checkbox.toggled.connect(self._on_detection_option_changed)

        btn_col.addWidget(self.import_btn)
        btn_col.addWidget(self.surface_checkbox)
        btn_col.addWidget(self.ball_checkbox)
        btn_col.addWidget(self.detect_btn)
        btn_col.addWidget(self.quit_btn)
        bottom.addLayout(btn_col)

        root.addLayout(bottom)
        self._update_detect_button_state()

    @staticmethod
    def _create_image_label(message):
        """원본과 결과 이미지에 공통으로 쓰는 QLabel을 만든다."""
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid black;")
        label.setFixedHeight(400)
        return label

    # ---------------- Import ----------------
    def on_import(self):
        dialog = QFileDialog(self, "Import", DEFAULT_DIR)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        dialog.setLabelText(QFileDialog.Accept, "Choose")

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected = [Path(p) for p in dialog.selectedFiles()]
        if not selected:
            return

        # 폴더 하나를 선택(진입하지 않고 Choose)한 경우 -> 폴더 내 이미지 전체
        if len(selected) == 1 and selected[0].is_dir():
            paths = sorted(
                p for p in selected[0].iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )
            if not paths:
                QMessageBox.warning(self, "Import", "폴더에 이미지가 없습니다.")
                return
            self._set_image_list(paths)
            return

        # 파일을 선택한 경우 -> 선택한 파일만
        files = [p for p in selected if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
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

        # 체크된 컨투어가 있으면 이미지 이동 시 자동으로 새 이미지를 검출한다.
        if self._has_detection_option():
            self._detect_current_image()
        else:
            self._clear_result("컨투어 항목을 하나 이상 선택하세요.")
            self._show_current_info()

    def on_detect(self):
        """현재 체크박스 설정으로 현재 이미지를 검출한다."""
        if not self.image_paths:
            QMessageBox.information(self, "Detect", "먼저 이미지를 불러오세요.")
            return
        self._detect_current_image()

    def _detect_current_image(self):
        """컨투어 모듈을 호출하고 우측 결과 화면을 갱신한다."""
        path = self.image_paths[self.current_index]
        try:
            result_image, result = create_detection_visualization(
                path,
                detect_surface=self.surface_checkbox.isChecked(),
                detect_balls=self.ball_checkbox.isChecked(),
            )
        except ValueError as error:
            self._clear_result("검출에 실패했습니다.")
            QMessageBox.critical(self, "Detect", str(error))
            return

        self._show_result_image(result_image)
        self._update_info_label(path, result)

    def _show_result_image(self, bgr_image):
        """OpenCV BGR 이미지를 QPixmap으로 바꿔 우측에 표시한다."""
        height, width, _ = bgr_image.shape
        qimage = QImage(
            bgr_image.data,
            width,
            height,
            bgr_image.strides[0],
            QImage.Format_BGR888,
        )
        # QImage가 NumPy 배열을 계속 참조하지 않도록 복사한다.
        self.result_pixmap = QPixmap.fromImage(qimage.copy())
        self._set_scaled_pixmap(self.result_label, self.result_pixmap)

    @staticmethod
    def _set_scaled_pixmap(label, pixmap):
        """라벨 크기 안에서 원본 비율을 유지해 이미지를 표시한다."""
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _has_detection_option(self):
        return self.surface_checkbox.isChecked() or self.ball_checkbox.isChecked()

    def _on_detection_option_changed(self):
        """아무 컨투어도 선택하지 않았을 때 Detect 버튼을 비활성화한다."""
        self._update_detect_button_state()
        if not self._has_detection_option():
            self._clear_result("컨투어 항목을 하나 이상 선택하세요.")

    def _update_detect_button_state(self):
        self.detect_btn.setEnabled(self._has_detection_option())

    def _clear_result(self, message):
        self.result_pixmap = QPixmap()
        self.result_label.setPixmap(QPixmap())
        self.result_label.setText(message)

    def _update_info_label(self, path, result=None):
        """파일 정보와 마지막 검출 개수를 하단에 표시한다."""
        lines = [
            f"파일명 : {path}",
            f"이미지 사이즈 : {self.original_pixmap.width()} x {self.original_pixmap.height()}",
        ]
        if result is not None:
            lines.append(f"표면 : {int(result.surface is not None)}")
            lines.append(f"볼의 개수 : {len(result.balls)}")
            for index, ball in enumerate(result.balls, start=1):
                lines.append(
                    f" - {index}번째 볼의 사이즈 : 반지름 {ball.radius}px"
                )
        self.info_label.setText("\n".join(lines))

    def _show_current_info(self):
        path = self.image_paths[self.current_index]
        self._update_info_label(path)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.show_prev()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
