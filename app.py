import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_DIR = "~" 


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Detector")
        self.resize(1000, 650)

        self.image_paths = []   # 불러온 이미지 경로 리스트
        self.current_index = -1

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- 이미지 표시 영역 ---
        self.image_label = QLabel("Import Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedHeight(400)
        root.addWidget(self.image_label)

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
        self.import_btn.clicked.connect(self.on_import)
        self.quit_btn.clicked.connect(self.close)
        btn_col.addWidget(self.import_btn)
        btn_col.addWidget(self.detect_btn)
        btn_col.addWidget(self.quit_btn)
        bottom.addLayout(btn_col)

        root.addLayout(bottom)

    # ---------------- Import ----------------
    def on_import(self):
        dialog = QFileDialog(self, "Import", DEFAULT_DIR)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
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


# 네비게이션

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
        pixmap = QPixmap(str(path))
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.index_label.setText(f"{self.current_index + 1} / {len(self.image_paths)}")
        self.info_label.setText(
            f"File Name : {path}\nImage Size : {pixmap.width()} x {pixmap.height()}"
        )
    
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