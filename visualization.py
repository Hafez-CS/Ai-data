import logging
from PyQt5.QtWidgets import QMessageBox

logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

class DataVisualizer:
    def __init__(self, parent):
        self.parent = parent

    # تابع visualize_data حذف شده زیرا نیازی به نمودارهای اولیه نیست
    def visualize_data(self):
        pass

    def show_scatter_plot(self):
        pass

    def compare_columns(self):
        pass