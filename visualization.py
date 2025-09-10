import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

    def visualize_data(self):
        if self.parent.df is None:
            return

        try:
            self.parent.figure.clear()
            numeric_cols = self.parent.df.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                self.parent.figure.set_size_inches(14, 12)
                ax1 = self.parent.figure.add_subplot(221)
                if len(numeric_cols) >= 2:
                    ax1.scatter(self.parent.df[numeric_cols[0]], self.parent.df[numeric_cols[1]], alpha=0.7)
                    ax1.set_xlabel(numeric_cols[0])
                    ax1.set_ylabel(numeric_cols[1])
                    ax1.set_title(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
                    ax1.tick_params(axis='x', rotation=45)

                ax2 = self.parent.figure.add_subplot(222)
                hist_cols = numeric_cols[:2]
                if hist_cols:
                    for i, col in enumerate(hist_cols):
                        ax2.hist(self.parent.df[col], bins=20, alpha=0.5, label=col, density=True)
                    ax2.legend()
                    ax2.set_title("Histogram ستون‌ها")
                    ax2.tick_params(axis='x', rotation=45)

                ax3 = self.parent.figure.add_subplot(223)
                if hist_cols:
                    self.parent.df[hist_cols].boxplot(ax=ax3)
                    ax3.set_title("Box Plot ستون‌ها")
                    ax3.tick_params(axis='x', rotation=45)

                ax4 = self.parent.figure.add_subplot(224)
                sns.heatmap(self.parent.df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
                ax4.set_title("Heatmap همبستگی")

                self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
                self.parent.canvas.draw()
                self.parent.status_bar.showMessage("نمودارهای اولیه تولید شدند.")
            else:
                logging.error("هیچ ستون عددی برای نمایش نمودار یافت نشد.")
                self.parent.status_bar.showMessage("هیچ ستون عددی برای نمایش نمودار یافت نشد.")
        except Exception as e:
            logging.error(f"خطا در تولید نمودارهای اولیه: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در تولید نمودارها: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در تولید نمودارها: {str(e)}")
    
    def show_scatter_plot(self):
        if self.parent.df is None:
            QMessageBox.warning(self.parent, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.parent.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return
    
        try:
            x_column = self.parent.combo_target.currentText()
            y_columns = [item.text() for item in self.parent.list_compare.selectedItems()]

            if not x_column or not y_columns:
                QMessageBox.warning(self.parent, "هشدار", "لطفاً یک ستون برای محور X و حداقل یک ستون برای محور Y انتخاب کنید.")
                self.parent.status_bar.showMessage("ستون‌های کافی انتخاب نشده‌اند.")
                return
            
            numeric_cols = self.parent.df.select_dtypes(include=np.number).columns.tolist()
            if x_column not in numeric_cols or not all(col in numeric_cols for col in y_columns):
                QMessageBox.warning(self.parent, "هشدار", "همه ستون‌های انتخاب‌شده باید عددی باشند.")
                self.parent.status_bar.showMessage("ستون‌های غیرعددی انتخاب شده‌اند.")
                return
        
            self.parent.figure.clear()
            self.parent.figure.set_size_inches(14, 12)
            ax = self.parent.figure.add_subplot(111)

            for y_column in y_columns:
                ax.scatter(self.parent.df[x_column], self.parent.df[y_column], alpha=0.7, label=f"{y_column} vs {x_column}")
            

            ax.set_xlabel(x_column)
            ax.set_ylabel("مقادیر")
            ax.set_title(f"نمودار پراکندگی: {x_column} در برابر ستون‌های انتخاب‌شده")
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)

            self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
            self.parent.canvas.draw()
            self.parent.status_bar.showMessage(f"نمودار پراکندگی برای {x_column} و {', '.join(y_columns)} تولید شد.")



        except Exception as e:
            logging.error(f"خطا در تولید نمودار پراکندگی: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در تولید نمودار پراکندگی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در تولید نمودار پراکندگی: {str(e)}")


    def compare_columns(self):
        try:
            selected_columns = [item.text() for item in self.parent.list_compare.selectedItems()]
            if len(selected_columns) not in [2, 3]:
                logging.error(f"تعداد ستون‌های انتخاب‌شده نامعتبر است: {len(selected_columns)} ستون انتخاب شده است.")
                QMessageBox.warning(self.parent, "هشدار", "لطفاً دقیقاً ۲ یا ۳ ستون را با Ctrl+کلیک انتخاب کنید.")
                self.parent.status_bar.showMessage("انتخاب نامعتبر: لطفاً ۲ یا ۳ ستون انتخاب کنید.")
                return
    
            numeric_cols = self.parent.df.select_dtypes(include=np.number).columns.tolist()
            if not all(col in numeric_cols for col in selected_columns):
                logging.error("ستون‌های غیرعددی برای مقایسه انتخاب شده‌اند.")
                QMessageBox.warning(self.parent, "هشدار", "همه ستون‌های انتخاب‌شده باید عددی باشند.")
                self.parent.status_bar.showMessage("ستون‌های غیرعددی انتخاب شده‌اند.")
                return
    
            self.parent.figure.clear()
            self.parent.figure.set_size_inches(14, 12)
            if len(selected_columns) == 2:
                ax = self.parent.figure.add_subplot(111)
                ax.scatter(self.parent.df[selected_columns[0]], self.parent.df[selected_columns[1]], alpha=0.7)
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title(f"مقایسه: {selected_columns[0]} vs {selected_columns[1]}")
                ax.tick_params(axis='x', rotation=45)
            else:
                ax = self.parent.figure.add_subplot(111, projection='3d')
                ax.scatter(self.parent.df[selected_columns[0]], self.parent.df[selected_columns[1]], self.parent.df[selected_columns[2]], alpha=0.7)
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_zlabel(selected_columns[2])
                ax.set_title(f"مقایسه 3D: {selected_columns[0]} vs {selected_columns[1]} vs {selected_columns[2]}")
                ax.tick_params(axis='x', rotation=45)
    
            self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
            self.parent.canvas.draw()
            self.parent.status_bar.showMessage(f"مقایسه ستون‌ها: {', '.join(selected_columns)}")
        except Exception as e:
            logging.error(f"خطا در مقایسه ستون‌ها: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در مقایسه ستون‌ها: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در مقایسه ستون‌ها: {str(e)}")