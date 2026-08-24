APP_STYLESHEET = """
    /* Toss Design System: Grey 50/100/200/600/700/900, Blue 50/500/600 */
    QMainWindow, #centralWidget {
        background: #F2F4F6;
        color: #191F28;
        font-family: Pretendard, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        font-size: 14px;
    }
    QLabel#windowTitle {
        color: #191F28; font-size: 28px; font-weight: 700; letter-spacing: -0.7px;
    }
    QLabel#headerMetadata { color: #8B95A1; font-size: 14px; }
    QLabel#indexLabel {
        color: #6B7684; font-size: 13px; min-width: 42px; qproperty-alignment: AlignCenter;
    }
    QFrame#imageCard, QFrame#analysisCard, QFrame#logCard, QFrame#controlCard {
        background: #FFFFFF; border: 1px solid #E5E8EB; border-radius: 20px;
    }
    QLabel#cardTitle, QLabel#sectionTitle {
        color: #333D4B; font-size: 15px; font-weight: 700;
    }
    QLabel#imagePaneTitle {
        color: #6B7684; font-size: 13px; font-weight: 700; padding-left: 2px;
    }
    QLabel#previewTitle {
        color: #4E5968; font-size: 13px; font-weight: 700; padding-left: 2px;
    }
    QFrame#settingRow {
        background: #F7F8FA; border: 1px solid #E5E8EB; border-radius: 10px;
    }
    QLabel#controlLabel { color: #4E5968; font-size: 13px; font-weight: 600; }
    QLabel#settingValue {
        color: #333D4B; background: #FFFFFF; border: 1px solid #E5E8EB;
        border-radius: 7px; font-size: 13px; font-weight: 600; padding: 5px 8px;
    }
    QFrame#ballCountSegment {
        background: #E5E8EB; border-radius: 8px;
    }
    QPushButton#ballCountSegmentButton {
        background: transparent; border: 0; border-radius: 6px; color: #6B7684;
        font-size: 13px; font-weight: 700; min-height: 28px; min-width: 30px; padding: 0 5px;
    }
    QPushButton#ballCountSegmentButton:hover { color: #3182F6; }
    QPushButton#ballCountSegmentButton:checked { background: #FFFFFF; color: #3182F6; }
    QLabel#imagePreview {
        background: #F9FAFB; border: 1px solid #E5E8EB; border-radius: 12px;
        color: #8B95A1; padding: 14px;
    }
    QLabel#infoLabel { color: #4E5968; font-size: 12px; line-height: 1.45; }
    QFrame#statusCard {
        background: #F7FAFF; border: 1px solid #D8E9FF; border-radius: 14px;
    }
    QLabel#statusValue { color: #3182F6; font-size: 22px; font-weight: 700; }
    QLabel#statusDetail { color: #6B7684; font-size: 12px; }
    QDialog#imageModal {
        background: #F2F4F6; border: 1px solid #D1D6DB; border-radius: 20px;
    }
    QFrame#imageModalCard { background: #FFFFFF; border-radius: 14px; }
    QLabel#imageModalTitle { color: #333D4B; font-size: 15px; font-weight: 700; }
    QLabel#imageModalPreview {
        background: #F9FAFB; border: 1px solid #E5E8EB; border-radius: 10px;
    }
    QLabel#imageModalHint { color: #8B95A1; font-size: 12px; }
    QScrollArea#imageModalScroll {
        background: #F9FAFB; border: 1px solid #E5E8EB; border-radius: 10px;
    }
    QScrollArea#imageModalScroll > QWidget > QWidget { background: #F9FAFB; }
    QPushButton#zoomButton, QPushButton#zoomResetButton {
        background: #F2F4F6; border: 0; border-radius: 8px; color: #4E5968;
        font-size: 13px; font-weight: 700; min-height: 28px; min-width: 30px; padding: 0 8px;
    }
    QPushButton#zoomButton:hover, QPushButton#zoomResetButton:hover {
        background: #E5E8EB; color: #3182F6;
    }
    QLabel#zoomLabel { color: #6B7684; font-size: 12px; min-width: 42px; qproperty-alignment: AlignCenter; }
    QCheckBox { color: #333D4B; font-size: 14px; spacing: 9px; }
    QCheckBox::indicator {
        width: 18px; height: 18px; border: 1px solid #D1D6DB;
        border-radius: 5px; background: #FFFFFF;
    }
    QCheckBox::indicator:hover { border-color: #90C2FF; }
    QCheckBox::indicator:checked { background: #3182F6; border-color: #3182F6; }
    QFrame#divider { color: #E5E8EB; max-height: 1px; }
    QPushButton {
        border: 0; border-radius: 12px; font-size: 14px; font-weight: 700;
        min-height: 38px; padding: 0 16px;
    }
    QPushButton#primaryButton { background: #3182F6; color: #FFFFFF; }
    QPushButton#primaryButton:hover { background: #2272EB; }
    QPushButton#primaryButton:pressed { background: #1B64DA; }
    QPushButton#primaryButton:disabled { background: #E5E8EB; color: #B0B8C1; }
    QPushButton#secondaryButton { background: #E8F3FF; color: #3182F6; }
    QPushButton#secondaryButton:hover { background: #C9E2FF; }
    QPushButton#secondaryButton:pressed { background: #90C2FF; color: #1957C2; }
    QPushButton#navigationButton {
        background: transparent; border: 0; border-radius: 12px; color: #4E5968;
        font-size: 23px; font-weight: 400; min-height: 30px; min-width: 30px; padding: 0;
    }
    QPushButton#navigationButton:hover { background: #E5E8EB; color: #191F28; }
    QPushButton#navigationButton:disabled { color: #D1D6DB; }
    QPushButton#captureIconButton {
        background: #E8F3FF; border: 0; border-radius: 10px;
        min-height: 32px; min-width: 32px; max-height: 32px; max-width: 32px;
        padding: 0;
    }
    QPushButton#captureIconButton:hover { background: #C9E2FF; }
    QPushButton#captureIconButton:pressed { background: #90C2FF; }
"""
