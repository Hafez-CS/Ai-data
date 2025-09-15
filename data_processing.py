import pandas as pd
import numpy as np
import logging
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from dialog import DataAnalysisDialog

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

class DataProcessor:
    def __init__(self, parent):
        self.parent = parent

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self.parent, "انتخاب فایل CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.parent.df = pd.read_csv(file_path)
                self.parent.list_compare.clear()
                self.parent.list_compare.addItems(self.parent.df.columns.tolist())
                self.parent.status_bar.showMessage("فایل با موفقیت بارگذاری شد!")
                QMessageBox.information(self.parent, "موفقیت", "فایل با موفقیت بارگذاری شد!")
                self.parent.visualizer.visualize_data()
                
                # خودکار کردن پاک‌سازی و داده‌کاوی بعد از بارگذاری
                self.clean_data()
                self.mine_data()
                
            except Exception as e:
                logging.error(f"خطا در بارگذاری فایل CSV: {str(e)}", exc_info=True)
                self.parent.status_bar.showMessage(f"خطا در بارگذاری فایل: {str(e)}")
                QMessageBox.critical(self.parent, "خطا", f"خطا در بارگذاری فایل: {str(e)}")

    def clean_data(self):
        if self.parent.df is None:
            QMessageBox.warning(self.parent, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.parent.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        try:
            df_cleaned = self.parent.df.copy()
            numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
            non_numeric_cols = df_cleaned.select_dtypes(exclude=np.number).columns

            # پر کردن مقادیر گمشده برای ستون‌های عددی
            if not numeric_cols.empty:
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
            
            # پر کردن مقادیر گمشده برای ستون‌های غیرعددی
            if not non_numeric_cols.empty:
                for col in non_numeric_cols:
                    mode_value = df_cleaned[col].mode()
                    if not mode_value.empty:  # بررسی اینکه مد وجود دارد
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('')  # در صورت نبود مد، با رشته خالی پر می‌کنیم

            # حذف داده‌های پرت برای ستون‌های عددی
            initial_rows = len(df_cleaned)
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

            self.parent.df = df_cleaned
            self.parent.list_compare.clear()
            self.parent.list_compare.addItems(self.parent.df.columns.tolist())

            cleaning_report = (
                f"تعداد ردیف‌های اولیه: {initial_rows}\n"
                f"تعداد ردیف‌های پس از پاک‌سازی: {len(df_cleaned)}\n"
                f"مقادیر گمشده پرشده با میانگین (ستون‌های عددی) و مد یا رشته خالی (ستون‌های غیرعددی)\n"
                f"داده‌های پرت حذف شدند (روش IQR برای ستون‌های عددی)"
            )
            dialog = DataAnalysisDialog("گزارش پاک‌سازی داده‌ها", cleaning_report, self.parent)
            dialog.exec_()
            self.parent.status_bar.showMessage("پاک‌سازی داده‌ها با موفقیت انجام شد.")
            self.parent.visualizer.visualize_data()
        except Exception as e:
            logging.error(f"خطا در پاک‌سازی داده‌ها: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در پاک‌سازی داده‌ها: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.parent.df is None:
            QMessageBox.warning(self.parent, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.parent.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        try:
            desc_stats = self.parent.df.describe(include='all').to_string()
            numeric_cols = self.parent.df.select_dtypes(include=np.number).columns
            corr_matrix = self.parent.df[numeric_cols].corr().to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
            outlier_report = ""
            for col in numeric_cols:
                Q1 = self.parent.df[col].quantile(0.25)
                Q3 = self.parent.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.parent.df[(self.parent.df[col] < lower_bound) | (self.parent.df[col] > upper_bound)][col]
                outlier_report += f"ستون {col}: {len(outliers)} داده پرت\n"

            mining_report = (
                "آمار توصیفی:\n" + desc_stats + "\n\n"
                "ماتریس همبستگی:\n" + corr_matrix + "\n\n"
                "گزارش داده‌های پرت:\n" + outlier_report
            )
            dialog = DataAnalysisDialog("نتایج داده‌کاوی", mining_report, self.parent)
            dialog.exec_()
            self.parent.status_bar.showMessage("داده‌کاوی با موفقیت انجام شد.")
        except Exception as e:
            logging.error(f"خطا در داده‌کاوی: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در داده‌کاوی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در داده‌کاوی: {str(e)}")