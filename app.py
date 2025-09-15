import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io
import os
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from google import genai
from google.genai import types
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import arabic_reshaper
from bidi.algorithm import get_display

# Configure logging
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# Configure Matplotlib for Persian text
font_path = "path/to/Vazir.ttf"  # Replace with actual path to Vazir font
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Vazir"
else:
    logging.warning("Vazir font not found. Falling back to default font.")

app = FastAPI(title="Advanced Financial Predictor API")

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_csv(self, file: UploadFile):
        try:
            self.df = pd.read_csv(file.file)
            return {"message": "فایل با موفقیت بارگذاری شد!", "columns": self.df.columns.tolist()}
        except Exception as e:
            logging.error(f"خطا در بارگذاری فایل CSV: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"خطا در بارگذاری فایل: {str(e)}")

    def clean_data(self):
        if self.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV را بارگذاری کنید.")
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

            cleaning_report = {
                "initial_rows": len(self.df),
                "cleaned_rows": len(df_cleaned),
                "message": "مقادیر گمشده پرشده با میانگین (ستون‌های عددی) و مد (ستون‌های غیرعددی). داده‌های پرت حذف شدند (روش IQR)."
            }
            self.df = df_cleaned
            return cleaning_report
        except Exception as e:
            logging.error(f"خطا در پاک‌سازی داده‌ها: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV را بارگذاری کنید.")
        try:
            desc_stats = self.df.describe().to_dict()
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            corr_matrix = self.df[numeric_cols].corr().to_dict()
            outlier_report = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                outlier_report[col] = len(outliers)

            return {
                "descriptive_stats": desc_stats,
                "correlation_matrix": corr_matrix,
                "outlier_report": outlier_report
            }
        except Exception as e:
            logging.error(f"خطا در داده‌کاوی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در داده‌کاوی: {str(e)}")

class DataVisualizer:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor

    def visualize_data(self):
        if self.data_processor.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV را بارگذاری کنید.")
        try:
            fig = plt.Figure(figsize=(14, 12))
            numeric_cols = self.data_processor.df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise HTTPException(status_code=400, detail="هیچ ستون عددی برای نمایش نمودار یافت نشد.")

            ax1 = fig.add_subplot(221)
            if len(numeric_cols) >= 2:
                ax1.scatter(self.data_processor.df[numeric_cols[0]], self.data_processor.df[numeric_cols[1]], alpha=0.7)
                ax1.set_xlabel(get_display(arabic_reshaper.reshape(numeric_cols[0])))
                ax1.set_ylabel(get_display(arabic_reshaper.reshape(numeric_cols[1])))
                ax1.set_title(get_display(arabic_reshaper.reshape(f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}")))
                ax1.tick_params(axis='x', rotation=45)

            ax2 = fig.add_subplot(222)
            hist_cols = numeric_cols[:2]
            if hist_cols:
                for col in hist_cols:
                    ax2.hist(self.data_processor.df[col], bins=20, alpha=0.5, label=get_display(arabic_reshaper.reshape(col)), density=True)
                ax2.legend()
                ax2.set_title(get_display(arabic_reshaper.reshape("Histogram ستون‌ها")))
                ax2.tick_params(axis='x', rotation=45)

            ax3 = fig.add_subplot(223)
            if hist_cols:
                self.data_processor.df[hist_cols].boxplot(ax=ax3)
                ax3.set_title(get_display(arabic_reshaper.reshape("Box Plot ستون‌ها")))
                ax3.tick_params(axis='x', rotation=45)

            ax4 = fig.add_subplot(224)
            sns.heatmap(self.data_processor.df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
            ax4.set_title(get_display(arabic_reshaper.reshape("Heatmap همبستگی")))

            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
            canvas = FigureCanvas(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            logging.error(f"خطا در تولید نمودارهای اولیه: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در تولید نمودارها: {str(e)}")

    def show_scatter_plot(self, x_column: str, y_columns: List[str]):
        if self.data_processor.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV را بارگذاری کنید.")
        try:
            numeric_cols = self.data_processor.df.select_dtypes(include=np.number).columns.tolist()
            if x_column not in numeric_cols or not all(col in numeric_cols for col in y_columns):
                raise HTTPException(status_code=400, detail="همه ستون‌های انتخاب‌شده باید عددی باشند.")
            fig = plt.Figure(figsize=(14, 12))
            ax = fig.add_subplot(111)
            for y_column in y_columns:
                ax.scatter(self.data_processor.df[x_column], self.data_processor.df[y_column], alpha=0.7, 
                          label=get_display(arabic_reshaper.reshape(f"{y_column} vs {x_column}")))
            ax.set_xlabel(get_display(arabic_reshaper.reshape(x_column)))
            ax.set_ylabel(get_display(arabic_reshaper.reshape("مقادیر")))
            ax.set_title(get_display(arabic_reshaper.reshape(f"نمودار پراکندگی: {x_column} در برابر ستون‌های انتخاب‌شده")))
            ax.legend()
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)
            canvas = FigureCanvas(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            logging.error(f"خطا در تولید نمودار پراکندگی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در تولید نمودار پراکندگی: {str(e)}")

    def compare_columns(self, selected_columns: List[str]):
        if self.data_processor.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV را بارگذاری کنید.")
        try:
            if len(selected_columns) not in [2, 3]:
                raise HTTPException(status_code=400, detail="لطفاً دقیقاً ۲ یا ۳ ستون انتخاب کنید.")
            numeric_cols = self.data_processor.df.select_dtypes(include=np.number).columns.tolist()
            if not all(col in numeric_cols for col in selected_columns):
                raise HTTPException(status_code=400, detail="همه ستون‌های انتخاب‌شده باید عددی باشند.")
            fig = plt.Figure(figsize=(14, 12))
            if len(selected_columns) == 2:
                ax = fig.add_subplot(111)
                ax.scatter(self.data_processor.df[selected_columns[0]], self.data_processor.df[selected_columns[1]], alpha=0.7)
                ax.set_xlabel(get_display(arabic_reshaper.reshape(selected_columns[0])))
                ax.set_ylabel(get_display(arabic_reshaper.reshape(selected_columns[1])))
                ax.set_title(get_display(arabic_reshaper.reshape(f"مقایسه: {selected_columns[0]} vs {selected_columns[1]}")))
                ax.tick_params(axis='x', rotation=45)
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(self.data_processor.df[selected_columns[0]], self.data_processor.df[selected_columns[1]], 
                          self.data_processor.df[selected_columns[2]], alpha=0.7)
                ax.set_xlabel(get_display(arabic_reshaper.reshape(selected_columns[0])))
                ax.set_ylabel(get_display(arabic_reshaper.reshape(selected_columns[1])))
                ax.set_zlabel(get_display(arabic_reshaper.reshape(selected_columns[2])))
                ax.set_title(get_display(arabic_reshaper.reshape(f"مقایسه 3D: {selected_columns[0]} vs {selected_columns[1]} vs {selected_columns[2]}")))
                ax.tick_params(axis='x', rotation=45)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3, hspace=0.4)
            canvas = FigureCanvas(fig)
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            logging.error(f"خطا در مقایسه ستون‌ها: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در مقایسه ستون‌ها: {str(e)}")

class Predictor:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
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

    def train_and_predict(self, target_column: str):
        if self.data_processor.df is None or not target_column:
            raise HTTPException(status_code=400, detail="لطفاً فایل CSV را بارگذاری کرده و ستون هدف را انتخاب کنید.")
        try:
            # بررسی نوع داده‌های عددی (شامل اعشاری)
            if self.data_processor.df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                logging.error("ستون هدف غیرعددی انتخاب شده است.")
                raise HTTPException(status_code=400, detail="ستون هدف باید عددی (صحیح یا اعشاری) باشد.")
            
            X = self.data_processor.df.drop(columns=[target_column])
            y = self.data_processor.df[target_column]

            # انتخاب ستون‌های عددی (شامل اعشاری)
            X = X.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32])
            if X.empty:
                logging.error("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                raise HTTPException(status_code=400, detail="هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
            
            if len(X) < 2 or len(y) < 2:
                logging.error("داده‌های کافی برای آموزش مدل وجود ندارد.")
                raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")
            
            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            if len(X_test) == 0 or len(y_test) == 0:
                logging.error("داده‌های آزمایشی کافی نیست.")
                raise HTTPException(status_code=400, detail="داده‌های آزمایشی کافی نیست.")
            
            # تحلیل دیتاست با Gemini API
            recommended_model, recommendation_text = self.analyze_dataset_with_gemini(self.data_processor.df, target_column)
            if not recommended_model:
                logging.error(f"Gemini نتوانست الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")
                raise HTTPException(status_code=500, detail=f"Gemini نتوانست الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")
            
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
                raise HTTPException(status_code=400, detail=f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")

            # آموزش و پیش‌بینی فقط با مدل پیشنهادی Gemini
            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # تولید داده‌های آینده‌نگرانه (فرضی)
                future_indices = np.arange(len(y_test), len(y_test) + 5)
                future_X = X_test[-5:]  # استفاده از آخرین داده‌های آزمایشی برای پیش‌بینی آینده
                future_pred = model.predict(future_X)

                # تولید گراف خطی
                fig = plt.Figure(figsize=(14, 8))
                ax = fig.add_subplot(111)
                indices = np.arange(len(y_test))
                
                # خطوط برای مقادیر واقعی و پیش‌بینی‌شده
                ax.plot(indices, y_test.values, color='blue', label=get_display(arabic_reshaper.reshape('مقادیر واقعی')), linewidth=2)
                ax.plot(indices, y_pred, color='orange', label=get_display(arabic_reshaper.reshape('مقادیر پیش‌بینی‌شده')), linewidth=2)
                
                # خط برای پیش‌بینی‌های آینده
                ax.plot(future_indices, future_pred, color='green', linestyle='--', 
                        label=get_display(arabic_reshaper.reshape('پیش‌بینی آینده')), linewidth=2)
                
                ax.set_xlabel(get_display(arabic_reshaper.reshape("اندیس داده‌ها")))
                ax.set_ylabel(get_display(arabic_reshaper.reshape("مقادیر")))
                ax.set_title(get_display(arabic_reshaper.reshape(f"پیش‌بینی {target_column} با مدل {recommended_model}")))
                ax.legend()
                ax.grid(True)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
                
                # ذخیره نمودار
                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                
                # ذخیره فایل برای دسترسی بعدی
                filename = f"prediction_{target_column}.png"
                with open(filename, "wb") as f:
                    f.write(buf.getvalue())

                return {
                    "message": f"پیش‌بینی با مدل {recommended_model} انجام شد.",
                    "gemini_recommendation": recommendation_text,
                    "plot": f"/plot/{filename}"
                }

            except Exception as e:
                logging.error(f"خطا در آموزش مدل {recommended_model}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"خطا در آموزش مدل {recommended_model}: {str(e)}")

        except Exception as e:
            logging.error(f"خطا در فرآیند پیش‌بینی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در فرآیند پیش‌بینی: {str(e)}")

data_processor = DataProcessor()
data_visualizer = DataVisualizer(data_processor)
predictor = Predictor(data_processor)

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV بارگذاری کنید.")
    return data_processor.load_csv(file)

@app.post("/clean_data")
async def clean_data():
    return data_processor.clean_data()

@app.get("/mine_data")
async def mine_data():
    return data_processor.mine_data()

@app.get("/visualize_data")
async def visualize_data():
    return data_visualizer.visualize_data()

@app.post("/scatter_plot")
async def scatter_plot(x_column: str = Form(...), y_columns: List[str] = Form(...)):
    return data_visualizer.show_scatter_plot(x_column, y_columns)

@app.post("/compare_columns")
async def compare_columns(selected_columns: List[str] = Form(...)):
    return data_visualizer.compare_columns(selected_columns)

@app.post("/predict")
async def predict(target_column: str = Form(...)):
    return predictor.train_and_predict(target_column)

@app.get("/plot/{filename}")
async def get_plot(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="نمودار یافت نشد.")
    with open(filename, "rb") as f:
        return StreamingResponse(io.BytesIO(f.read()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)