import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QPushButton, QFileDialog, QLabel,
    QComboBox, QMessageBox, QTableWidget, QTableWidgetItem, QListWidget,
    QStatusBar, QScrollArea, QVBoxLayout, QDialog, QTextEdit
)
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

try:
    import xgboost as xgb
except ImportError:
    xgb = None

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

class AdvancedFinancialPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("پیش‌بینی مالی پیشرفته")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.grid_layout.addWidget(self.status_bar, 7, 0, 1, 3)

        # File loading section
        self.btn_load = QPushButton("بارگذاری فایل CSV")
        self.btn_load.clicked.connect(self.load_csv)
        self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.btn_load, 0, 0, 1, 1)

        # Target column selection
        self.label_target = QLabel("ستون هدف:")
        self.combo_target = QComboBox()
        self.grid_layout.addWidget(self.label_target, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.combo_target, 0, 2, 1, 1)

        # Column comparison selection
        self.label_compare = QLabel("ستون‌ها برای مقایسه (۲ یا ۳ ستون):")
        self.list_compare = QListWidget()
        self.list_compare.setSelectionMode(QListWidget.MultiSelection)
        self.btn_compare = QPushButton("مقایسه ستون‌ها")
        self.btn_compare.clicked.connect(self.compare_columns)
        self.btn_compare.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.label_compare, 1, 0, 1, 1)
        self.grid_layout.addWidget(self.list_compare, 1, 1, 1, 2)
        self.grid_layout.addWidget(self.btn_compare, 2, 0, 1, 1)

        # Data cleaning button
        self.btn_clean = QPushButton("پاک‌سازی داده‌ها")
        self.btn_clean.clicked.connect(self.clean_data)
        self.btn_clean.setStyleSheet("background-color: #FFC107; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.btn_clean, 2, 1, 1, 1)

        # Data mining button
        self.btn_mine = QPushButton("داده‌کاوی")
        self.btn_mine.clicked.connect(self.mine_data)
        self.btn_mine.setStyleSheet("background-color: #9C27B0; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.btn_mine, 2, 2, 1, 1)

        # Data scatter button
        self.btn_scatter = QPushButton("نمایش پراکندگی داده‌ها")
        self.btn_scatter.clicked.connect(self.show_scatter_plot)
        self.btn_scatter.setStyleSheet("background-color: #00BCD4; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.btn_scatter, 3, 0, 1, 1)

        # Model selection
        self.label_model = QLabel("مدل‌های پیش‌بینی:")
        self.combo_model = QComboBox()
        models = [
            "Linear Regression",
            "Random Forest Regressor",
            "Decision Tree Regressor",
            "Gradient Boosting Regressor",
            "Support Vector Regressor"
        ]
        if xgb:
            models.append("XGBoost Regressor")
        self.combo_model.addItems(models)
        self.grid_layout.addWidget(self.label_model, 3, 1, 1, 1)
        self.grid_layout.addWidget(self.combo_model, 3, 2, 1, 1)

        # Predict button
        self.btn_predict = QPushButton("شروع پیش‌بینی")
        self.btn_predict.clicked.connect(self.train_and_predict)
        self.btn_predict.setStyleSheet("background-color: #FF9800; color: white; padding: 8px;")
        self.grid_layout.addWidget(self.btn_predict, 4, 0, 1, 1)

        # Results table
        self.table_results = QTableWidget()
        self.grid_layout.addWidget(self.table_results, 4, 1, 1, 2)

        # Scroll area for Matplotlib canvas
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(600)
        self.grid_layout.addWidget(self.scroll_area, 5, 0, 2, 3)

        self.df = None
        self.status_bar.showMessage("لطفاً فایل CSV را بارگذاری کنید.")

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget { font-size: 14px; }
            QPushButton { border-radius: 5px; }
            QComboBox, QListWidget { padding: 5px; }
            QScrollArea { border: 1px solid #ccc; background-color: #f9f9f9; }
        """)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.combo_target.clear()
                self.list_compare.clear()
                self.combo_target.addItems(self.df.columns.tolist())
                self.list_compare.addItems(self.df.columns.tolist())
                self.status_bar.showMessage("فایل با موفقیت بارگذاری شد!")
                QMessageBox.information(self, "موفقیت", "فایل با موفقیت بارگذاری شد!")
                self.visualize_data()
            except Exception as e:
                self.status_bar.showMessage(f"خطا در بارگذاری فایل: {str(e)}")
                QMessageBox.critical(self, "خطا", f"خطا در بارگذاری فایل: {str(e)}")

    def clean_data(self):
        if self.df is None:
            QMessageBox.warning(self, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        try:
            df_cleaned = self.df.copy()
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

            self.df = df_cleaned
            self.combo_target.clear()
            self.list_compare.clear()
            self.combo_target.addItems(self.df.columns.tolist())
            self.list_compare.addItems(self.df.columns.tolist())

            cleaning_report = (
                f"تعداد ردیف‌های اولیه: {len(self.df)}\n"
                f"تعداد ردیف‌های پس از پاک‌سازی: {len(df_cleaned)}\n"
                f"مقادیر گمشده پرشده با میانگین (ستون‌های عددی) و مد (ستون‌های غیرعددی)\n"
                f"داده‌های پرت حذف شدند (روش IQR)"
            )
            dialog = DataAnalysisDialog("گزارش پاک‌سازی داده‌ها", cleaning_report, self)
            dialog.exec_()
            self.status_bar.showMessage("پاک‌سازی داده‌ها با موفقیت انجام شد.")
            self.visualize_data()
        except Exception as e:
            self.status_bar.showMessage(f"خطا در پاک‌سازی داده‌ها: {str(e)}")
            QMessageBox.critical(self, "خطا", f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.df is None:
            QMessageBox.warning(self, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        try:
            desc_stats = self.df.describe().to_string()
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            corr_matrix = self.df[numeric_cols].corr().to_string()
            outlier_report = ""
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                outlier_report += f"ستون {col}: {len(outliers)} داده پرت\n"

            mining_report = (
                "آمار توصیفی:\n" + desc_stats + "\n\n"
                "ماتریس همبستگی:\n" + corr_matrix + "\n\n"
                "گزارش داده‌های پرت:\n" + outlier_report
            )
            dialog = DataAnalysisDialog("نتایج داده‌کاوی", mining_report, self)
            dialog.exec_()
            self.status_bar.showMessage("داده‌کاوی با موفقیت انجام شد.")
        except Exception as e:
            self.status_bar.showMessage(f"خطا در داده‌کاوی: {str(e)}")
            QMessageBox.critical(self, "خطا", f"خطا در داده‌کاوی: {str(e)}")

    def visualize_data(self):
        if self.df is None:
            return

        self.figure.clear()
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            self.figure.set_size_inches(14, 12)
            ax1 = self.figure.add_subplot(221)
            if len(numeric_cols) >= 2:
                ax1.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], alpha=0.7)
                ax1.set_xlabel(numeric_cols[0])
                ax1.set_ylabel(numeric_cols[1])
                ax1.set_title(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
                ax1.tick_params(axis='x', rotation=45)

            ax2 = self.figure.add_subplot(222)
            hist_cols = numeric_cols[:2]
            if hist_cols:
                for i, col in enumerate(hist_cols):
                    ax2.hist(self.df[col], bins=20, alpha=0.5, label=col, density=True)
                ax2.legend()
                ax2.set_title("Histogram ستون‌ها")
                ax2.tick_params(axis='x', rotation=45)

            ax3 = self.figure.add_subplot(223)
            if hist_cols:
                self.df[hist_cols].boxplot(ax=ax3)
                ax3.set_title("Box Plot ستون‌ها")
                ax3.tick_params(axis='x', rotation=45)

            ax4 = self.figure.add_subplot(224)
            sns.heatmap(self.df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
            ax4.set_title("Heatmap همبستگی")

            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
            self.canvas.draw()
            self.status_bar.showMessage("نمودارهای اولیه تولید شدند.")

    def show_scatter_plot(self):
        if self.df is None:
            QMessageBox.warning(self, "هشدار", "لطفاً ابتدا فایل CSV را بارگذاری کنید.")
            self.status_bar.showMessage("هیچ فایلی بارگذاری نشده است.")
            return

        self.figure.clear()
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "هشدار", "حداقل دو ستون عددی برای نمایش پراکندگی مورد نیاز است.")
            self.status_bar.showMessage("ستون‌های عددی کافی نیست.")
            return

        self.figure.set_size_inches(14, 12)
        plot_cols = numeric_cols[:4]
        g = sns.pairplot(self.df[plot_cols], diag_kind="kde", plot_kws={"alpha": 0.7})
        self.figure = g.figure
        self.canvas = FigureCanvas(self.figure)
        self.scroll_area.setWidget(self.canvas)
        self.canvas.draw()
        self.status_bar.showMessage("نمودار پراکندگی داده‌ها تولید شد.")

    def compare_columns(self):
        selected_columns = [item.text() for item in self.list_compare.selectedItems()]
        if len(selected_columns) not in [2, 3]:
            QMessageBox.warning(self, "هشدار", "لطفاً ۲ یا ۳ ستون انتخاب کنید.")
            self.status_bar.showMessage("انتخاب نامعتبر: ۲ یا ۳ ستون مورد نیاز است.")
            return

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not all(col in numeric_cols for col in selected_columns):
            QMessageBox.warning(self, "هشدار", "همه ستون‌های انتخاب‌شده باید عددی باشند.")
            self.status_bar.showMessage("ستون‌های غیرعددی انتخاب شده‌اند.")
            return

        self.figure.clear()
        self.figure.set_size_inches(14, 12)
        if len(selected_columns) == 2:
            ax = self.figure.add_subplot(111)
            ax.scatter(self.df[selected_columns[0]], self.df[selected_columns[1]], alpha=0.7)
            ax.set_xlabel(selected_columns[0])
            ax.set_ylabel(selected_columns[1])
            ax.set_title(f"مقایسه: {selected_columns[0]} vs {selected_columns[1]}")
            ax.tick_params(axis='x', rotation=45)
        else:
            ax = self.figure.add_subplot(111, projection='3d')
            ax.scatter(self.df[selected_columns[0]], self.df[selected_columns[1]], self.df[selected_columns[2]], alpha=0.7)
            ax.set_xlabel(selected_columns[0])
            ax.set_ylabel(selected_columns[1])
            ax.set_zlabel(selected_columns[2])
            ax.set_title(f"مقایسه 3D: {selected_columns[0]} vs {selected_columns[1]} vs {selected_columns[2]}")
            ax.tick_params(axis='x', rotation=45)

        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
        self.canvas.draw()
        self.status_bar.showMessage(f"مقایسه ستون‌ها: {', '.join(selected_columns)}")

    def train_and_predict(self):
        if self.df is None or not self.combo_target.currentText():
            QMessageBox.warning(self, "هشدار", "لطفاً فایل CSV را بارگذاری کرده و ستون هدف را انتخاب کنید.")
            self.status_bar.showMessage("هیچ فایل یا ستون هدفی انتخاب نشده است.")
            return

        try:
            target_column = self.combo_target.currentText()
            # Check if target column is numeric
            if self.df[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                QMessageBox.critical(self, "خطا", "ستون هدف باید عددی باشد.")
                self.status_bar.showMessage("ستون هدف غیرعددی است.")
                return

            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]

            # Filter numeric columns for X
            X = X.select_dtypes(include=np.number)
            if X.empty:
                QMessageBox.critical(self, "خطا", "هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                self.status_bar.showMessage("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                return

            # Check for sufficient data
            if len(X) < 2 or len(y) < 2:
                QMessageBox.critical(self, "خطا", "داده‌های کافی برای آموزش مدل وجود ندارد.")
                self.status_bar.showMessage("داده‌های کافی نیست.")
                return

            # Handle missing values
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)

            # Normalize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Check if test set is empty
            if len(X_test) == 0 or len(y_test) == 0:
                QMessageBox.critical(self, "خطا", "داده‌های آزمایشی کافی نیست. لطفاً داده‌های بیشتری فراهم کنید.")
                self.status_bar.showMessage("داده‌های آزمایشی کافی نیست.")
                return

            # Select model
            model_name = self.combo_model.currentText()
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Decision Tree Regressor":
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == "Gradient Boosting Regressor":
                model = GradientBoostingRegressor(random_state=42)
            elif model_name == "Support Vector Regressor":
                model = SVR()
            elif model_name == "XGBoost Regressor" and xgb:
                model = xgb.XGBRegressor(random_state=42)
            else:
                QMessageBox.critical(self, "خطا", "مدل انتخاب‌شده پشتیبانی نمی‌شود.")
                self.status_bar.showMessage("مدل انتخاب‌شده پشتیبانی نمی‌شود.")
                return

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            QMessageBox.information(self, "نتایج مدل", f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")
            self.status_bar.showMessage(f"پیش‌بینی با {model_name} انجام شد.")

            # Display results in table
            self.table_results.setRowCount(len(y_test))
            self.table_results.setColumnCount(2)
            self.table_results.setHorizontalHeaderLabels(["مقادیر واقعی", "مقادیر پیش‌بینی شده"])
            for i in range(len(y_test)):
                self.table_results.setItem(i, 0, QTableWidgetItem(str(round(y_test.values[i], 2))))
                self.table_results.setItem(i, 1, QTableWidgetItem(str(round(y_pred[i], 2))))

            # Plot actual vs predicted
            self.figure.clear()
            self.figure.set_size_inches(14, 12)
            ax = self.figure.add_subplot(111)
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("مقادیر واقعی")
            ax.set_ylabel("مقادیر پیش‌بینی شده")
            ax.set_title(f"پیش‌بینی {target_column} با {model_name}")
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
            self.canvas.draw()

        except Exception as e:
            self.status_bar.showMessage(f"خطا در پیش‌بینی: {str(e)}")
            QMessageBox.critical(self, "خطا", f"خطا در فرآیند پیش‌بینی: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedFinancialPredictor()
    window.show()
    sys.exit(app.exec_())