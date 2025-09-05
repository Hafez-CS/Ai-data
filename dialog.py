from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

class DataAnalysisDialog(QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(text)
        layout.addWidget(self.text_edit)
        close_button = QPushButton("بستن")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        self.setLayout(layout)