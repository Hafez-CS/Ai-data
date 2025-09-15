from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QListWidget, QTableWidget, QScrollArea, QStatusBar, QFileDialog, 
    QMessageBox, QStyle
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from data_processing import DataProcessor
from visualization import DataVisualizer
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
            QLabel {
                color: #37474F;
                font-weight: bold;
            }
            QComboBox, QListWidget {
                border: 1px solid #BBDEFB;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
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

        # Initialize data processor, visualizer, and predictor
        self.data_processor = DataProcessor(self)
        self.visualizer = DataVisualizer(self)
        self.predictor = Predictor(self)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top panel for file operations
        top_panel = QHBoxLayout()
        file_group = QVBoxLayout()
        self.btn_load = QPushButton("بارگذاری فایل CSV")
        self.btn_load.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        file_group.addWidget(self.btn_load)
        
        target_group = QVBoxLayout()
        self.label_target = QLabel("ستون هدف:")
        self.combo_target = QComboBox()
        target_group.addWidget(self.label_target)
        target_group.addWidget(self.combo_target)

        top_panel.addLayout(file_group)
        top_panel.addLayout(target_group)
        main_layout.addLayout(top_panel)

        # Middle panel for analysis controls
        middle_panel = QHBoxLayout()
        column_group = QVBoxLayout()
        self.label_compare = QLabel("ستون‌های مقایسه:")
        self.list_compare = QListWidget()
        self.list_compare.setSelectionMode(QListWidget.ExtendedSelection)  # فعال کردن انتخاب چندگانه
        self.list_compare.setToolTip("برای انتخاب ۲ یا ۳ ستون، از Ctrl+کلیک استفاده کنید")
        self.list_compare.setMaximumHeight(150)
        column_group.addWidget(self.label_compare)
        column_group.addWidget(self.list_compare)

        buttons_group = QVBoxLayout()
        self.btn_scatter = QPushButton("نمودار پراکندگی")
        self.btn_compare = QPushButton("مقایسه ستون‌ها")
        
        buttons_group.addWidget(self.btn_scatter)
        buttons_group.addWidget(self.btn_compare)

        middle_panel.addLayout(column_group)
        middle_panel.addLayout(buttons_group)
        main_layout.addLayout(middle_panel)

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
        self.btn_scatter.clicked.connect(self.visualizer.show_scatter_plot)
        self.btn_compare.clicked.connect(self.visualizer.compare_columns)
        self.btn_predict.clicked.connect(self.predictor.train_and_predict)

        # Initialize
        self.df = None
        self.status_bar.showMessage("آماده برای بارگذاری فایل")