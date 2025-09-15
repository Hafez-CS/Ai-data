import sys
import logging
from PyQt5.QtWidgets import QApplication
from ui import AdvancedFinancialPredictorUI

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename='app_errors.log',  
    filemode='a',   
    format='%(asctime)s - %(levelname)s - %(message)s',  
    level=logging.ERROR  
)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = AdvancedFinancialPredictorUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error("خطا در اجرای برنامه", exc_info=True)
        print(f"خطا رخ داد: {str(e)}. جزئیات در فایل app_errors.log ذخیره شد.")