import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from dialog import DataAnalysisDialog

class DataProcessor:
    def __init__(self, parent):
        self.parent = parent

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self.parent, "انتخاب فایل CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.parent.df = pd.read_csv(file_path)
                self.parent.combo_target.clear()
                self.parent.list_compare.clear()
                self.parent.combo_target.addItems(self.parent.df.columns.tolist())
                self.parent.list_compare.addItems(self.parent.df.columns.tolist())
                self.parent.status_bar.showMessage("فایل با موفقیت بارگذاری شد!")
                QMessageBox.information(self.parent, "موفقیت", "فایل با موفقیت بارگذاری شد!")
                self.parent.visualizer.visualize_data()
            except Exception as e:
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
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
            non_numeric_cols = df_cleaned.select_dtypes(exclude=np.number).columns
            df_cleaned[non_numeric_cols] = df_cleaned[non_numeric_cols].fillna(df_cleaned[non_numeric_cols].mode().iloc[0])

            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

            self.parent.df = df_cleaned
            self.parent.combo_target.clear()
            self.parent.list_compare.clear()
            self.parent.combo_target.addItems(self.parent.df.columns.tolist())
            self.parent.list_compare.addItems(self.parent.df.columns.tolist())

            cleaning_report = (
                f"تعداد ردیف‌های اولیه: {len(self.parent.df)}\n"
                f"تعداد ردیف‌های پس از پاک‌سازی: {len(df_cleaned)}\n"
                f"مقادیر گمشده پرشده با میانگین (ستون‌های عددی) و مد (ستون‌های غیرعددی)\n"
                f"داده‌های پرت حذف شدند (روش IQR)"
            )
            dialog = DataAnalysisDialog("گزارش پاک‌سازی داده‌ها", cleaning_report, self.parent)
            dialog.exec_()
            self.parent.status_bar.showMessage("پاک‌سازی داده‌ها با موفقیت انجام شد.")
            self.parent.visualizer.visualize_data()
        except Exception as e:
            self.parent.status_bar.showMessage(f"خطا در پاک‌سازی داده‌ها: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.parent.df is None:
            QMessageBox.warning(self.parent, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.parent.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        try:
            desc_stats = self.parent.df.describe().to_string()
            numeric_cols = self.parent.df.select_dtypes(include=np.number).columns
            corr_matrix = self.parent.df[numeric_cols].corr().to_string()
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
            self.parent.status_bar.showMessage(f"خطا در داده‌کاوی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در داده‌کاوی: {str(e)}")