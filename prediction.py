import numpy as np
import logging
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

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

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
                logging.error("ستون هدف غیرعددی انتخاب شده است.")
                QMessageBox.critical(self.parent, "خطا", "ستون هدف باید عددی باشد.")
                self.parent.status_bar.showMessage("ستون هدف غیرعددی است.")
                return

            X = self.parent.df.drop(columns=[target_column])
            y = self.parent.df[target_column]

            X = X.select_dtypes(include=np.number)
            if X.empty:
                logging.error("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                QMessageBox.critical(self.parent, "خطا", "هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                self.parent.status_bar.showMessage("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                return

            if len(X) < 2 or len(y) < 2:
                logging.error("داده‌های کافی برای آموزش مدل وجود ندارد.")
                QMessageBox.critical(self.parent, "خطا", "داده‌های کافی برای آموزش مدل وجود ندارد.")
                self.parent.status_bar.showMessage("داده‌های کافی نیست.")
                return

            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
                logging.error("داده‌های آزمایشی کافی نیست.")
                QMessageBox.critical(self.parent, "خطا", "داده‌های آزمایشی کافی نیست. لطفاً داده‌های بیشتری فراهم کنید.")
                self.parent.status_bar.showMessage("داده‌های آزمایشی کافی نیست.")
                return

            # تعریف مدل‌ها
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }
            if xgb:
                models["XGBoost"] = xgb.XGBRegressor(random_state=42)

            # آزمایش همه مدل‌ها و انتخاب بهترین
            best_model_name = None
            best_r2 = -float('inf')
            best_model = None
            best_y_pred = None

            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = model_name
                        best_model = model
                        best_y_pred = y_pred
                except Exception as e:
                    logging.error(f"خطا در آموزش مدل {model_name}: {str(e)}", exc_info=True)
                    continue  # ادامه با مدل بعدی در صورت خطا

            if best_model is None:
                logging.error("هیچ مدلی با موفقیت آموزش ندید.")
                QMessageBox.critical(self.parent, "خطا", "هیچ مدلی با موفقیت آموزش ندید. لطفاً داده‌ها را بررسی کنید.")
                self.parent.status_bar.showMessage("خطا: هیچ مدلی آموزش ندید.")
                return

            # محاسبه معیارها برای بهترین مدل
            mae = mean_absolute_error(y_test, best_y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, best_y_pred))
            QMessageBox.information(self.parent, "نتایج مدل",
                                    f"بهترین مدل: {best_model_name}\n"
                                    f"MAE: {mae:.2f}\n"
                                    f"RMSE: {rmse:.2f}\n"
                                    f"R²: {best_r2:.2f}")
            self.parent.status_bar.showMessage(f"پیش‌بینی با بهترین مدل ({best_model_name}) انجام شد.")

            # نمایش نتایج در جدول
            self.parent.table_results.setRowCount(len(y_test))
            self.parent.table_results.setColumnCount(2)
            self.parent.table_results.setHorizontalHeaderLabels(["مقادیر واقعی", "مقادیر پیش‌بینی شده"])
            for i in range(len(y_test)):
                self.parent.table_results.setItem(i, 0, QTableWidgetItem(str(round(y_test.values[i], 2))))
                self.parent.table_results.setItem(i, 1, QTableWidgetItem(str(round(best_y_pred[i], 2))))

            # نمایش نمودار پراکندگی
            self.parent.figure.clear()
            self.parent.figure.set_size_inches(14, 12)
            ax = self.parent.figure.add_subplot(111)
            ax.scatter(y_test, best_y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("مقادیر واقعی")
            ax.set_ylabel("مقادیر پیش‌بینی شده")
            ax.set_title(f"پیش‌بینی {target_column} با {best_model_name}")
            self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
            self.parent.canvas.draw()

        except Exception as e:
            logging.error(f"خطا در فرآیند پیش‌بینی: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در پیش‌بینی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در فرآیند پیش‌بینی: {str(e)}")