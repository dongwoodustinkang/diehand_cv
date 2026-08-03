APP_STYLESHEET = """
    /* Toss Design System: Grey 50/100/200/600/700/900, Blue 50/500/600 */
    QMainWindow, #centralWidget {
        background: #F2F4F6;
        color: #191F28;
        font-family: Pretendard, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        font-size: 14px;
    }
    QLabel#windowTitle {
        color: #191F28; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;
    }
    QLabel#subtitle { color: #6B7684; font-size: 14px; }
    QLabel#indexLabel {
        color: #6B7684; font-size: 13px; min-width: 42px; qproperty-alignment: AlignCenter;
    }
    QFrame#imageCard, QFrame#infoCard, QFrame#controlCard {
        background: #FFFFFF; border: 1px solid #E5E8EB; border-radius: 16px;
    }
    QLabel#cardTitle, QLabel#sectionTitle {
        color: #333D4B; font-size: 15px; font-weight: 700;
    }
    QLabel#imagePreview {
        background: #F9FAFB; border: 1px solid #E5E8EB; border-radius: 12px;
        color: #8B95A1; padding: 20px;
    }
    QLabel#infoLabel { color: #4E5968; line-height: 1.45; }
    QCheckBox { color: #333D4B; font-size: 14px; spacing: 9px; }
    QCheckBox::indicator {
        width: 18px; height: 18px; border: 1px solid #D1D6DB;
        border-radius: 5px; background: #FFFFFF;
    }
    QCheckBox::indicator:hover { border-color: #90C2FF; }
    QCheckBox::indicator:checked { background: #3182F6; border-color: #3182F6; }
    QFrame#divider { color: #F2F4F6; max-height: 1px; }
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
"""
