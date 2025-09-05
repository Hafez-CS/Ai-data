import sys
from PyQt5.QtWidgets import QApplication
from ui import AdvancedFinancialPredictorUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedFinancialPredictorUI()
    window.show()
    sys.exit(app.exec_())