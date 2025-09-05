import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class Predictor:
    def __init__(self, parent):
        self.parent = parent

    def train_and_predict(self):
        if self.parent.df is None or not self.parent.combo_target.currentText():
            QMessageBox.warning(self.parent, "هشدار", "لطفاً فایل CSV را بارگذاری کرده و ستون هدف را انتخاب کنید.")
            self.parent.status_bar.showMessage("هیچ فایل یا ستون هدفی انتخاب نشده است.")
            return

        try:
            target_column = self.parent.combo_target.currentText()
            if self.parent.df[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                QMessageBox.critical(self.parent, "خطا", "ستون هدف باید عددی باشد.")
                self.parent.status_bar.showMessage("ستون هدف غیرعددی است.")
                return

            X = self.parent.df.drop(columns=[target_column])
            y = self.parent.df[target_column]

            X = X.select_dtypes(include=np.number)
            if X.empty:
                QMessageBox.critical(self.parent, "خطا", "هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                self.parent.status_bar.showMessage("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                return

            if len(X) < 2 or len(y) < 2:
                QMessageBox.critical(self.parent, "خطا", "داده‌های کافی برای آموزش مدل وجود ندارد.")
                self.parent.status_bar.showMessage("داده‌های کافی نیست.")
                return

            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
                QMessageBox.critical(self.parent, "خطا", "داده‌های آزمایشی کافی نیست. لطفاً داده‌های بیشتری فراهم کنید.")
                self.parent.status_bar.showMessage("داده‌های آزمایشی کافی نیست.")
                return

            model_name = self.parent.combo_model.currentText()
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(random_state=42)
            elif model_name == "SVR":
                model = SVR()
            elif model_name == "XGBoost" and xgb:
                model = xgb.XGBRegressor(random_state=42)
            else:
                QMessageBox.critical(self.parent, "خطا", "مدل انتخاب‌شده پشتیبانی نمی‌شود.")
                self.parent.status_bar.showMessage("مدل انتخاب‌شده پشتیبانی نمی‌شود.")
                return

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            QMessageBox.information(self.parent, "نتایج مدل", f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")
            self.parent.status_bar.showMessage(f"پیش‌بینی با {model_name} انجام شد.")

            self.parent.table_results.setRowCount(len(y_test))
            self.parent.table_results.setColumnCount(2)
            self.parent.table_results.setHorizontalHeaderLabels(["مقادیر واقعی", "مقادیر پیش‌بینی شده"])
            for i in range(len(y_test)):
                self.parent.table_results.setItem(i, 0, QTableWidgetItem(str(round(y_test.values[i], 2))))
                self.parent.table_results.setItem(i, 1, QTableWidgetItem(str(round(y_pred[i], 2))))

            self.parent.figure.clear()
            self.parent.figure.set_size_inches(14, 12)
            ax = self.parent.figure.add_subplot(111)
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("مقادیر واقعی")
            ax.set_ylabel("مقادیر پیش‌بینی شده")
            ax.set_title(f"پیش‌بینی {target_column} با {model_name}")
            self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
            self.parent.canvas.draw()

        except Exception as e:
            self.parent.status_bar.showMessage(f"خطا در پیش‌بینی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در فرآیند پیش‌بینی: {str(e)}")