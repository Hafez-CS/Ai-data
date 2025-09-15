from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QScrollArea, QStatusBar, QFileDialog, 
    QMessageBox, QStyle
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from data_processing import DataProcessor
from prediction import Predictor

class AdvancedFinancialPredictorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("سیستم پیش‌بینی مالی")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'B Nazanin';
                font-size: 12px;
                background-color: #f0f2f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                min-width: 120px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #BBDEFB;
                border-radius: 4px;
            }
            QScrollArea {
                border: 1px solid #BBDEFB;
                background-color: white;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #E3F2FD;
                color: #1565C0;
            }
        """)

        # Initialize data processor and predictor
        self.data_processor = DataProcessor(self)
        self.predictor = Predictor(self)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top panel for file operations
        top_panel = QHBoxLayout()
        file_group = QVBoxLayout()
        self.btn_load = QPushButton("بارگذاری فایل CSV یا Excel")
        self.btn_load.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        file_group.addWidget(self.btn_load)
        top_panel.addLayout(file_group)
        main_layout.addLayout(top_panel)

        # Model selection and prediction panel
        model_panel = QHBoxLayout()
        self.btn_predict = QPushButton("شروع پیش‌بینی")
        self.btn_predict.setStyleSheet("background-color: #4CAF50;")
        model_panel.addWidget(self.btn_predict)
        main_layout.addLayout(model_panel)

        # Results panel
        results_panel = QHBoxLayout()
        self.table_results = QTableWidget()
        self.table_results.setMinimumHeight(200)
        
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        
        results_panel.addWidget(self.table_results, 1)
        results_panel.addWidget(self.scroll_area, 2)
        main_layout.addLayout(results_panel)

        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        # Connect signals
        self.btn_load.clicked.connect(self.data_processor.load_csv)
        self.btn_predict.clicked.connect(self.predictor.train_and_predict)

        # Initialize
        self.df = None
        self.status_bar.showMessage("آماده برای بارگذاری فایل")