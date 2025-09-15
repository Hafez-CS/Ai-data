import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import QMessageBox
import os
from google import genai
from google.genai import types
import matplotlib.pyplot as plt

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
        # تنظیم API Key برای Gemini
        os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'AIzaSyBJNUay3LtA_3vjU_M0sayBbQQ0xpdGclY')
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = "gemini-2.5-pro"

    def analyze_dataset_with_gemini(self, df, target_column):
        """تحلیل دیتاست با Gemini API و پیشنهاد بهترین الگوریتم"""
        try:
            # آماده‌سازی مشخصات دیتاست
            numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            desc_stats = df[numeric_cols].describe().to_string()
            corr_matrix = df[numeric_cols].corr().to_string()
            num_rows, num_cols = df.shape
            missing_values = df.isnull().sum().sum()

            # ساخت پرامپت برای Gemini
            prompt = f"""
            شما یک متخصص یادگیری ماشین هستید. من یک دیتاست با مشخصات زیر دارم:
            - تعداد ردیف‌ها: {num_rows}
            - تعداد ستون‌ها: {num_cols}
            - ستون‌های عددی: {list(numeric_cols)}
            - ستون هدف: {target_column}
            - آمار توصیفی:
            {desc_stats}
            - ماتریس همبستگی:
            {corr_matrix}
            - تعداد مقادیر گمشده: {missing_values}

            با توجه به این اطلاعات، بهترین الگوریتم یادگیری ماشین برای رگرسیون را از بین گزینه‌های زیر پیشنهاد دهید:
            Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, XGBoost (اگر موجود باشد).
            لطفاً فقط نام الگوریتم را به صورت دقیق (مثلاً 'Random Forest') و توضیح مختصری برای پیشنهاد خود ارائه دهید.
            """

            # تنظیم محتوا برای Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                ),
            )

            # دریافت پاسخ از Gemini
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text

            # استخراج نام الگوریتم از پاسخ
            recommended_model = None
            available_models = ["Linear Regression", "Random Forest", "Decision Tree", 
                               "Gradient Boosting", "SVR"]
            if xgb:
                available_models.append("XGBoost")
            for model_name in available_models:
                if model_name.lower() in response_text.lower():
                    recommended_model = model_name
                    break

            return recommended_model, response_text

        except Exception as e:
            logging.error(f"خطا در تحلیل دیتاست با Gemini API: {str(e)}", exc_info=True)
            return None, f"خطا در تحلیل دیتاست با Gemini API: {str(e)}"

    def train_and_predict(self):
        if self.parent.df is None or not self.parent.combo_target.currentText():
            QMessageBox.warning(self.parent, "هشدار", "لطفاً فایل CSV را بارگذاری کرده و ستون هدف را انتخاب کنید.")
            self.parent.status_bar.showMessage("هیچ فایل یا ستون هدفی انتخاب نشده است.")
            return

        try:
            target_column = self.parent.combo_target.currentText()
            # بررسی نوع داده‌های عددی (شامل اعشاری)
            if self.parent.df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                logging.error("ستون هدف غیرعددی انتخاب شده است.")
                QMessageBox.critical(self.parent, "خطا", "ستون هدف باید عددی (صحیح یا اعشاری) باشد.")
                self.parent.status_bar.showMessage("ستون هدف غیرعددی است.")
                return

            X = self.parent.df.drop(columns=[target_column])
            y = self.parent.df[target_column]

            # انتخاب ستون‌های عددی (شامل اعشاری)
            X = X.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32])
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

            # تحلیل دیتاست با Gemini API
            recommended_model, recommendation_text = self.analyze_dataset_with_gemini(self.parent.df, target_column)
            if not recommended_model:
                logging.error(f"Gemini نتوانست الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")
                QMessageBox.critical(self.parent, "خطا", 
                                   f"Gemini نتوانست الگوریتم مناسبی پیشنهاد دهد:\n{recommendation_text}")
                self.parent.status_bar.showMessage("Gemini نتوانست الگوریتم پیشنهاد دهد.")
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

            # بررسی معتبر بودن مدل پیشنهادی
            if recommended_model not in models:
                logging.error(f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")
                QMessageBox.critical(self.parent, "خطا", 
                                   f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")
                self.parent.status_bar.showMessage(f"الگوریتم {recommended_model} در دسترس نیست.")
                return

            # نمایش توصیه Gemini
            QMessageBox.information(self.parent, "توصیه Gemini", 
                                  f"مدل انتخاب‌شده: {recommended_model}\nتوضیحات: {recommendation_text}")
            self.parent.status_bar.showMessage(f"پیش‌بینی با مدل {recommended_model} انجام شد.")

            # آموزش و پیش‌بینی فقط با مدل پیشنهادی Gemini
            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # پاک کردن جدول نتایج
                self.parent.table_results.setRowCount(0)

                # تولید داده‌های آینده‌نگرانه (فرضی)
                # فرض می‌کنیم ۵ نقطه فراتر از داده‌های آزمایشی برای پیش‌بینی آینده
                future_indices = np.arange(len(y_test), len(y_test) + 5)
                future_X = X_test[-5:]  # استفاده از آخرین داده‌های آزمایشی برای پیش‌بینی آینده
                future_pred = model.predict(future_X)

                # نمایش نمودار خطی
                self.parent.figure.clear()
                self.parent.figure.set_size_inches(14, 8)
                ax = self.parent.figure.add_subplot(111)
                indices = np.arange(len(y_test))
                
                # خطوط برای مقادیر واقعی و پیش‌بینی‌شده
                ax.plot(indices, y_test.values, color='blue', label='مقادیر واقعی', linewidth=2)
                ax.plot(indices, y_pred, color='orange', label='مقادیر پیش‌بینی‌شده', linewidth=2)
                
                # خط برای پیش‌بینی‌های آینده
                ax.plot(future_indices, future_pred, color='green', linestyle='--', label='پیش‌بینی آینده', linewidth=2)
                
                ax.set_xlabel("اندیس داده‌ها")
                ax.set_ylabel("مقادیر")
                ax.set_title(f"پیش‌بینی {target_column} با مدل {recommended_model}")
                ax.legend()
                ax.grid(True)
                self.parent.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
                self.parent.canvas.draw()

            except Exception as e:
                logging.error(f"خطا در آموزش مدل {recommended_model}: {str(e)}", exc_info=True)
                QMessageBox.critical(self.parent, "خطا", f"خطا در آموزش مدل {recommended_model}: {str(e)}")
                self.parent.status_bar.showMessage(f"خطا در آموزش مدل {recommended_model}")
                return

        except Exception as e:
            logging.error(f"خطا در فرآیند پیش‌بینی: {str(e)}", exc_info=True)
            self.parent.status_bar.showMessage(f"خطا در پیش‌بینی: {str(e)}")
            QMessageBox.critical(self.parent, "خطا", f"خطا در فرآیند پیش‌بینی: {str(e)}")