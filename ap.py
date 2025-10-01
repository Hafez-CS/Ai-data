import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import arabic_reshaper
from bidi.algorithm import get_display
import json
import pyodbc
import os
import re
from uuid import uuid4

app = FastAPI()

# فعال‌سازی CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Access the API key
api_key = "AIzaSyB8Rz8vHUO0ASP90_QF7VR9pvkXYWgfH_I"  # Replace with your actual API key
if not api_key:
    raise ValueError("GEMINI_API_KEY not found or is empty")

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logging.warning("کتابخانه xgboost نصب نشده است. XGBoost در دسترس نخواهد بود.")

# Setup logging
logging.basicConfig(
    filename='app_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
    level=logging.DEBUG
)

# Database connection setup
DB_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=AI;"
    "UID=sa;"
    "PWD=123;"
)
try:
    conn = pyodbc.connect(DB_CONNECTION_STRING)
    logging.debug("اتصال به پایگاه داده با موفقیت برقرار شد.")
    cursor = conn.cursor()
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.columns WHERE Name = N'status' AND Object_ID = Object_ID(N'dbo.result_table'))
        ALTER TABLE dbo.result_table ADD status VARCHAR(50)
    """)
    conn.commit()
except Exception as e:
    logging.error(f"خطا در اتصال به پایگاه داده: {str(e)}", exc_info=True)
    raise Exception(f"خطا در اتصال به پایگاه داده: {str(e)}")

font_path = "C:/path/to/Vazir.ttf"  # مسیر فونت را به‌روزرسانی کنید
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Vazir"
else:
    logging.warning("فونت Vazir یافت نشد. از فونت پیش‌فرض استفاده می‌شود.")
    plt.rcParams["font.family"] = "sans-serif"

class DataProcessor:
    def __init__(self):
        self.df = None
        self.ai_id = None

    async def load_file(self, file: UploadFile, filename: str, chat_id: int):
        logging.debug(f"بارگذاری فایل: {filename} برای chat_id: {chat_id}")
        try:
            file_content = await file.read()
            file_content_b64 = base64.b64encode(file_content).decode('utf-8')
            file_type = 'csv' if filename.endswith('.csv') else 'excel'
            logging.debug("محتوای فایل به صورت base64 کدگذاری شد.")

            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO dbo.ai (file_path, chat_id, filename, file_type, status)
                VALUES (?, ?, ?, ?, 'processed')
                """,
                (file_content_b64, chat_id, filename, file_type)
            )
            conn.commit()
            self.ai_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]
            logging.debug(f"محتوای فایل در جدول dbo.ai با ai_id {self.ai_id} ذخیره شد.")

            file_like = io.BytesIO(file_content)
            if filename.endswith('.csv'):
                self.df = pd.read_csv(file_like)
                logging.debug("فایل CSV با موفقیت خوانده شد.")
            elif filename.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_like)
                logging.debug("فایل Excel با موفقیت خوانده شد.")
            else:
                logging.error("فرمت فایل پشتیبانی نمی‌شود.")
                raise ValueError("فرمت فایل پشتیبانی نمی‌شود. فقط فایل‌های CSV و Excel مجاز هستند.")

            if self.df.empty:
                logging.error("فایل بارگذاری‌شده خالی است.")
                raise ValueError("فایل بارگذاری‌شده خالی است.")

            numeric_cols = self.df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            logging.debug(f"ستون‌های عددی اولیه: {numeric_cols.tolist()}")

            logging.debug("شروع پاک‌سازی داده‌ها")
            clean_report = self.clean_data()
            logging.debug("شروع داده‌کاوی")
            mine_report = self.mine_data()

            logging.info(f"فایل {filename} با موفقیت بارگذاری و پردازش شد.")
            return {
                "message": "فایل با موفقیت بارگذاری شد!",
                "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
                "cleaning_report": clean_report,
                "mining_report": mine_report,
                "chat_id": chat_id,
                "ai_id": int(self.ai_id)
            }
        except Exception as e:
            if self.ai_id:
                cursor.execute("UPDATE dbo.ai SET status = 'failed' WHERE id = ?", (self.ai_id,))
                conn.commit()
            logging.error(f"خطا در بارگذاری فایل {filename}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در بارگذاری فایل: {str(e)}")

    async def load_file_from_db(self, chat_id: int):
        logging.debug(f"بازیابی فایل برای chat_id: {chat_id}")
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, file_type FROM dbo.ai WHERE chat_id = ? AND status = 'processed' ORDER BY created_at DESC", (chat_id,))
            result = cursor.fetchone()
            if not result:
                logging.error(f"هیچ فایل پردازش‌شده‌ای برای chat_id {chat_id} یافت نشد.")
                raise HTTPException(status_code=404, detail=f"هیچ فایل پردازش‌شده‌ای برای chat_id {chat_id} یافت نشد.")

            self.ai_id, file_content_b64, file_type = result
            file_content = base64.b64decode(file_content_b64)
            file_like = io.BytesIO(file_content)

            if file_type == 'csv':
                self.df = pd.read_csv(file_like)
                logging.debug("فایل CSV از پایگاه داده خوانده شد.")
            elif file_type == 'excel':
                self.df = pd.read_excel(file_like)
                logging.debug("فایل Excel از پایگاه داده خوانده شد.")
            else:
                logging.error("فرمت فایل پشتیبانی نمی‌شود.")
                raise HTTPException(status_code=400, detail="فرمت فایل پشتیبانی نمی‌شود.")

            if self.df.empty:
                logging.error("فایل بارگذاری‌شده از پایگاه داده خالی است.")
                raise HTTPException(status_code=400, detail="فایل بارگذاری‌شده از پایگاه داده خالی است.")

            numeric_cols = self.df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            logging.debug(f"ستون‌های عددی پس از بارگذاری: {numeric_cols.tolist()}")

            return self.df
        except Exception as e:
            logging.error(f"خطا در بازیابی فایل برای chat_id {chat_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در بازیابی فایل: {str(e)}")

    def clean_data(self):
        if self.df is None:
            logging.error("هیچ داده‌ای برای پاک‌سازی وجود ندارد.")
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            df_cleaned = self.df.copy()
            logging.debug(f"شروع پاک‌سازی داده‌ها با {len(df_cleaned)} ردیف و {len(df_cleaned.columns)} ستون")

            columns_before = df_cleaned.columns.tolist()
            df_cleaned = df_cleaned.dropna(axis=1, how='all')
            columns_after = df_cleaned.columns.tolist()
            dropped_columns = set(columns_before) - set(columns_after)
            if dropped_columns:
                logging.debug(f"ستون‌های حذف‌شده (کاملاً خالی): {dropped_columns}")

            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
            logging.debug(f"ستون‌های عددی: {numeric_cols.tolist()}")
            logging.debug(f"ستون‌های غیرعددی: {non_numeric_cols.tolist()}")

            if not numeric_cols.empty:
                imputer = KNNImputer(n_neighbors=5)
                df_cleaned[numeric_cols] = pd.DataFrame(imputer.fit_transform(df_cleaned[numeric_cols]), columns=numeric_cols)
                logging.debug("مقادیر گمشده ستون‌های عددی با KNN Imputer پر شدند.")

            if not non_numeric_cols.empty:
                for col in non_numeric_cols:
                    mode_value = df_cleaned[col].mode()
                    if not mode_value.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna('')
                    logging.debug(f"مقادیر گمشده ستون غیرعددی {col} با مد یا رشته خالی پر شدند.")

            for col in non_numeric_cols:
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if not df_cleaned[col].isna().all():
                        numeric_cols = numeric_cols.append(pd.Index([col]))
                        logging.debug(f"ستون {col} به نوع عددی تبدیل شد.")
                except:
                    logging.debug(f"ستون {col} قابل تبدیل به عدد نیست و به صورت غیرعددی باقی می‌ماند.")

            initial_rows = len(df_cleaned)
            for col in numeric_cols:
                if col in df_cleaned.columns and df_cleaned[col].var() > 0:
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if np.isnan(IQR) or np.isinf(IQR):
                        logging.debug(f"IQR برای ستون {col} نامعتبر است، از حذف پرت‌ها صرف‌نظر شد.")
                        continue
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    logging.debug(f"داده‌های پرت برای ستون {col} حذف شدند.")
                else:
                    logging.debug(f"ستون {col} واریانس صفر دارد یا وجود ندارد و از حذف پرت‌ها صرف‌نظر شد.")

            if len(df_cleaned) < 2 or df_cleaned[numeric_cols].dropna().empty:
                logging.error("پس از پاک‌سازی، داده‌های کافی یا ستون‌های عددی معتبر باقی نمانده است.")
                raise HTTPException(status_code=400, detail="پس از پاک‌سازی، داده‌های کافی یا ستون‌های عددی معتبر باقی نمانده است.")

            self.df = df_cleaned
            logging.info(f"پاک‌سازی داده‌ها با موفقیت انجام شد. ردیف‌های اولیه: {initial_rows}, ردیف‌های نهایی: {len(df_cleaned)}")
            return {
                "initial_rows": initial_rows,
                "cleaned_rows": len(df_cleaned),
                "numeric_columns": numeric_cols.tolist(),
                "non_numeric_columns": non_numeric_cols.tolist(),
                "dropped_columns": list(dropped_columns),
                "message": "مقادیر گمشده با KNN Imputer (ستون‌های عددی) و مد یا رشته خالی (ستون‌های غیرعددی) پر شدند. داده‌های پرت حذف شدند (روش IQR)."
            }
        except Exception as e:
            logging.error(f"خطا در پاک‌سازی داده‌ها: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در پاک‌سازی داده‌ها: {str(e)}")

    def mine_data(self):
        if self.df is None:
            logging.error("هیچ داده‌ای برای داده‌کاوی وجود ندارد.")
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            desc_stats = self.df.describe(include='all')
            desc_stats = desc_stats.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
            logging.debug("آمار توصیفی محاسبه و مقادیر نامعتبر جایگزین شدند.")

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                corr_matrix = self.df[numeric_cols].corr()
                corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
                logging.debug("ماتریس همبستگی محاسبه و مقادیر نامعتبر جایگزین شدند.")
            else:
                corr_matrix = {}
                logging.debug("هیچ ستون عددی برای محاسبه ماتریس همبستگی وجود ندارد.")

            outlier_report = {}
            for col in numeric_cols:
                if self.df[col].var() > 0:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if np.isnan(IQR) or np.isinf(IQR):
                        outlier_report[col] = 0
                        logging.debug(f"IQR برای ستون {col} نامعتبر است، تعداد پرت‌ها صفر در نظر گرفته شد.")
                        continue
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                    outlier_report[col] = len(outliers)
                else:
                    outlier_report[col] = 0
                    logging.debug(f"ستون {col} واریانس صفر دارد، تعداد پرت‌ها صفر است.")

            logging.info("داده‌کاوی با موفقیت انجام شد.")
            return {
                "descriptive_stats": desc_stats,
                "correlation_matrix": corr_matrix,
                "outlier_report": outlier_report
            }
        except Exception as e:
            logging.error(f"خطا در داده‌کاوی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در داده‌کاوی: {str(e)}")

class Predictor:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        try:
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.5-pro"
            logging.debug("اتصال به Gemini API با موفقیت برقرار شد.")
        except Exception as e:
            logging.error(f"خطا در اتصال به Gemini API: {str(e)}", exc_info=True)
            raise Exception(f"خطا در اتصال به Gemini API: {str(e)}")

    async def analyze_dataset_with_gemini(self, df, target_column=None):
        logging.debug("شروع تحلیل دیتاست با Gemini API")
        try:
            sample_size = min(500, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            logging.debug(f"نمونه‌برداری انجام شد: {sample_size} ردیف")

            numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            all_cols = df.columns.tolist()
            desc_stats = df_sample.describe(include='all').replace([np.inf, -np.inf], np.nan).fillna(0).to_string()
            corr_matrix = df_sample[numeric_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0).to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
            num_rows, num_cols = df.shape
            missing_values = df.isnull().sum().sum()

            if target_column:
                if target_column not in df.columns:
                    logging.error(f"ستون هدف {target_column} در دیتاست وجود ندارد.")
                    raise HTTPException(status_code=400, detail=f"ستون هدف {target_column} در دیتاست وجود ندارد.")
                if df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                    logging.error(f"ستون هدف {target_column} غیرعددی است.")
                    raise HTTPException(status_code=400, detail=f"ستون هدف {target_column} باید عددی باشد.")

                prompt = f"""
                You are a machine learning expert. I have a dataset sample with the following details (sample size: {sample_size} rows):
                - Total rows in dataset: {num_rows}
                - Number of columns: {num_cols}
                - All columns: {all_cols}
                - Numeric columns: {numeric_cols.tolist()}
                - Descriptive statistics (based on sample):
                {desc_stats}
                - Correlation matrix (for numeric columns in sample):
                {corr_matrix}
                - Total missing values in dataset: {missing_values}
                - Selected target column: {target_column}

                Based on the dataset characteristics and the selected target column ({target_column}):
                Recommend the best machine learning algorithm for regression from the following options:
                Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, XGBoost (if available).
                Provide the algorithm name in the format 'model: <model_name>' and a brief explanation for the recommendation.
                """
            else:
                prompt = f"""
                You are a machine learning expert. I have a dataset sample with the following details (sample size: {sample_size} rows):
                - Total rows in dataset: {num_rows}
                - Number of columns: {num_cols}
                - All columns: {all_cols}
                - Numeric columns: {numeric_cols.tolist()}
                - Descriptive statistics (based on sample):
                {desc_stats}
                - Correlation matrix (for numeric columns in sample):
                {corr_matrix}
                - Total missing values in dataset: {missing_values}

                Based on this information:
                1. Recommend the best column to use as the target column for regression from the numeric columns: {numeric_cols.tolist()}.
                The target column must be numeric and selected based on correlation, variance, or predictive importance.
                2. Recommend the best machine learning algorithm for regression from the following options:
                Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, XGBoost (if available).
                Provide the target column and algorithm names in the format 'target_column: <column_name>' and 'model: <model_name>', along with a brief explanation for each recommendation.
                """

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            )

            response_text = ""
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text is not None:
                        response_text += chunk.text
                    else:
                        logging.warning("تکه پاسخ Gemini None است، نادیده گرفته شد.")
                        continue
                logging.debug(f"پاسخ کامل Gemini: {response_text}")
            except Exception as e:
                logging.error(f"خطا در دریافت پاسخ از Gemini API: {str(e)}", exc_info=True)
                recommended_target = target_column or next((col for col in df.columns if col in df.select_dtypes(include=[np.number]).columns), None)
                recommended_model = "Linear Regression"
                return recommended_target, recommended_model, f"خطا در دریافت پاسخ از Gemini API: {str(e)}. مقدار پیش‌فرض انتخاب شد."

            if not response_text:
                logging.error("پاسخ Gemini خالی است.")
                recommended_target = target_column or next((col for col in df.columns if col in df.select_dtypes(include=[np.number]).columns), None)
                recommended_model = "Linear Regression"
                return recommended_target, recommended_model, "پاسخ Gemini خالی بود، اولین ستون عددی و Linear Regression به‌صورت پیش‌فرض انتخاب شدند."

            recommended_target = target_column
            recommended_model = None
            available_models = ["Linear Regression", "Random Forest", "Decision Tree",
                               "Gradient Boosting", "SVR"]
            if xgb:
                available_models.append("XGBoost")

            model_match = re.search(r"model: ([\w\s]+)", response_text, re.IGNORECASE)
            recommended_model = model_match.group(1) if model_match and model_match.group(1) in available_models else None

            if not recommended_model:
                recommended_model = "Linear Regression"
                logging.debug("مدل پیش‌فرض Linear Regression انتخاب شد.")

            if recommended_target and recommended_model:
                logging.info(f"Gemini ستون هدف {recommended_target} و مدل {recommended_model} را پیشنهاد داد.")
                return recommended_target, recommended_model, response_text
            else:
                logging.error(f"Gemini نتوانست مدل مناسبی پیشنهاد دهد: {response_text}")
                recommended_target = target_column or next((col for col in df.columns if col in df.select_dtypes(include=[np.number]).columns), None)
                recommended_model = "Linear Regression"
                return recommended_target, recommended_model, f"Gemini نتوانست مدل پیشنهاد دهد: {response_text}. Linear Regression انتخاب شد."

        except Exception as e:
            logging.error(f"خطا در تحلیل دیتاست با Gemini API: {str(e)}", exc_info=True)
            recommended_target = target_column or next((col for col in df.columns if col in df.select_dtypes(include=[np.number]).columns), None)
            recommended_model = "Linear Regression"
            return recommended_target, recommended_model, f"خطا در تحلیل دیتاست با Gemini API: {str(e)}. مقدار پیش‌فرض انتخاب شد."

    async def train_and_predict(self, chat_id: int, target_column: Optional[str] = None):
        if self.data_processor.df is None:
            await self.data_processor.load_file_from_db(chat_id)
        try:
            logging.debug("شروع فرآیند پیش‌بینی")
            recommended_target, recommended_model, recommendation_text = await self.analyze_dataset_with_gemini(self.data_processor.df, target_column)
            if not recommended_target:
                logging.error(f"هیچ ستون عددی معتبری برای هدف یافت نشد: {recommendation_text}")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="هیچ ستون عددی معتبری برای هدف یافت نشد.")

            target_column = recommended_target
            logging.debug(f"ستون هدف: {target_column}, مدل: {recommended_model}")

            df_processed = pd.get_dummies(self.data_processor.df.drop(columns=[target_column], errors='ignore'), drop_first=True)
            df_processed[target_column] = self.data_processor.df[target_column]
            logging.debug(f"ستون‌های پس از get_dummies: {df_processed.columns.tolist()}")

            if target_column not in df_processed.columns:
                logging.error(f"ستون هدف {target_column} در داده‌های پردازش‌شده وجود ندارد.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail=f"ستون هدف {target_column} در داده‌های پردازش‌شده وجود ندارد.")

            if df_processed[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                logging.error(f"ستون هدف {target_column} غیرعددی است.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="ستون هدف پیشنهادی باید عددی باشد.")

            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]

            if X.empty:
                logging.error("هیچ ستون برای ویژگی‌ها یافت نشد.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="هیچ ستون برای ویژگی‌ها یافت نشد.")

            if len(X) < 2 or len(y) < 2:
                logging.error("داده‌های کافی برای آموزش مدل وجود ندارد.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                X_numeric = X[numeric_cols]
                vif_data = pd.DataFrame()
                vif_data["feature"] = X_numeric.columns
                vif_data["VIF"] = [variance_inflation_factor(X_numeric.fillna(0).values, i) for i in range(X_numeric.shape[1])]
                high_vif_cols = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
                if high_vif_cols:
                    logging.debug(f"حذف ستون‌های با VIF بالا: {high_vif_cols}")
                    X_numeric = X_numeric.drop(columns=high_vif_cols)
                X = X_numeric
            else:
                logging.error("هیچ ستون عددی برای ویژگی‌ها یافت نشد.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="هیچ ستون عددی برای ویژگی‌ها یافت نشد.")

            X = X.loc[:, X.var(numeric_only=True) > 0]
            X = X.loc[:, X.notna().any()]
            X.fillna(X.mean(numeric_only=True), inplace=True)
            y.fillna(y.mean(), inplace=True)

            if X.empty or len(X.columns) == 0:
                logging.error("پس از حذف ستون‌های نامعتبر، هیچ ویژگی برای آموزش باقی نماند.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="هیچ ویژگی معتبری برای آموزش مدل باقی نماند.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logging.error("داده‌های استانداردشده شامل مقادیر NaN یا بی‌نهایت هستند.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="داده‌های استانداردشده شامل مقادیر نامعتبر (NaN یا بی‌نهایت) هستند.")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
                logging.error("داده‌های آزمایشی کافی نیست.")
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                raise HTTPException(status_code=400, detail="داده‌های آزمایشی کافی نیست.")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }
            if xgb:
                models["XGBoost"] = xgb.XGBRegressor(random_state=42)

            if recommended_model not in models:
                logging.error(f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")
                recommended_model = "Linear Regression"
                logging.debug("مدل پیش‌فرض Linear Regression انتخاب شد.")

            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                logging.debug(f"مدل {recommended_model} با موفقیت آموزش دید.")

                future_X = X_test[-5:]
                future_pred = model.predict(future_X)
                logging.debug("پیش‌بینی‌های آینده تولید شدند.")

                fig = plt.Figure(figsize=(14, 8))
                ax = fig.add_subplot(111)
                indices = np.arange(len(y_test))

                ax.plot(indices, y_test.values, color='blue', label=get_display(arabic_reshaper.reshape('مقادیر واقعی')), linewidth=2)
                ax.plot(indices, y_pred, color='orange', label=get_display(arabic_reshaper.reshape('مقادیر پیش‌بینی‌شده')), linewidth=2)
                ax.plot(np.arange(len(y_test), len(y_test) + 5), future_pred, color='green', linestyle='--',
                        label=get_display(arabic_reshaper.reshape('پیش‌بینی آینده')), linewidth=2)

                ax.set_xlabel(get_display(arabic_reshaper.reshape("اندیس داده‌ها")))
                ax.set_ylabel(get_display(arabic_reshaper.reshape("مقادیر")))
                ax.set_title(get_display(arabic_reshaper.reshape(f"پیش‌بینی {target_column} با مدل {recommended_model}")))
                ax.legend()
                ax.grid(True)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                plot_data_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                logging.debug("نمودار به صورت base64 کدگذاری شد.")

                response_data = {
                    "message": f"پیش‌بینی با مدل {recommended_model} انجام شد.",
                    "target_column": target_column,
                    "model_used": recommended_model,
                    "gemini_recommendation": recommendation_text,
                    "plot_data": f"data:image/png;base64,{plot_data_b64}",
                    "prediction_data": {
                        "actual_values": y_test.tolist(),
                        "predicted_values": y_pred.tolist(),
                        "future_predictions": future_pred.tolist()
                    },
                    "chat_id": chat_id,
                    "ai_id": int(self.data_processor.ai_id)
                }

                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO dbo.result_table (chat_id, ai_id, result_json, model_used, target_column, plot_data, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'completed')
                    """,
                    (chat_id, self.data_processor.ai_id, json.dumps(response_data, ensure_ascii=False),
                     recommended_model, target_column, plot_data_b64)
                )
                conn.commit()
                logging.debug(f"نتایج پیش‌بینی برای chat_id {chat_id} و ai_id {self.data_processor.ai_id} در جدول dbo.result_table ذخیره شد.")

                logging.info(f"پیش‌بینی با مدل {recommended_model} برای ستون {target_column} با موفقیت انجام شد.")
                return response_data

            except Exception as e:
                cursor = conn.cursor()
                cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                               (chat_id, self.data_processor.ai_id))
                conn.commit()
                logging.error(f"خطا در آموزش مدل {recommended_model}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"خطا در آموزش مدل {recommended_model}: {str(e)}")

        except Exception as e:
            cursor = conn.cursor()
            cursor.execute("UPDATE dbo.result_table SET status = 'failed' WHERE chat_id = ? AND ai_id = ?",
                           (chat_id, self.data_processor.ai_id))
            conn.commit()
            logging.error(f"خطا در فرآیند پیش‌بینی: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"خطا در فرآیند پیش‌بینی: {str(e)}")

data_processor = DataProcessor()
predictor = Predictor(data_processor)

class PredictRequest(BaseModel):
    chat_id: int
    target_column: str  # ستون هدف اجباری است

@app.get("/")
async def index():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/upload_file")
async def upload_file(file: UploadFile, chat_id: int = Form(...)):
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            logging.error(f"فرمت فایل {file.filename} پشتیبانی نمی‌شود.")
            raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV یا Excel بارگذاری کنید.")

        response = await data_processor.load_file(file, file.filename, chat_id)
        return response
    except Exception as e:
        logging.error(f"خطا در آپلود فایل: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_numeric_columns/{chat_id}")
async def get_numeric_columns(chat_id: int):
    try:
        await data_processor.load_file_from_db(chat_id)
        numeric_cols = data_processor.df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns.tolist()
        if not numeric_cols:
            logging.error(f"هیچ ستون عددی برای chat_id {chat_id} یافت نشد.")
            raise HTTPException(status_code=400, detail="هیچ ستون عددی در دیتاست یافت نشد.")
        logging.debug(f"ستون‌های عددی برای chat_id {chat_id}: {numeric_cols}")
        return {"chat_id": chat_id, "numeric_columns": numeric_cols}
    except Exception as e:
        logging.error(f"خطا در بازیابی ستون‌های عددی برای chat_id {chat_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        response_data = await predictor.train_and_predict(request.chat_id)
        return response_data
    except Exception as e:
        logging.error(f"خطا در پیش‌بینی: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_target")
async def predict_with_target(request: PredictRequest):
    try:
        await data_processor.load_file_from_db(request.chat_id)
        # تمیز کردن نام ستون‌ها: حذف فاصله‌های اضافی
        data_processor.df.columns = data_processor.df.columns.str.strip()
        # گرفتن ستون‌های عددی
        numeric_cols = data_processor.df.select_dtypes(include=[np.number]).columns.str.strip().tolist()
        
        if not numeric_cols:
            logging.error(f"هیچ ستون عددی برای chat_id {request.chat_id} یافت نشد.")
            raise HTTPException(status_code=400, detail="هیچ ستون عددی در دیتاست یافت نشد.")
        
        if not request.target_column:
            logging.error(f"ستون هدف برای chat_id {request.chat_id} ارائه نشده است.")
            raise HTTPException(
                status_code=400,
                detail=f"لطفاً یک ستون هدف عددی ارائه کنید. ستون‌های عددی موجود: {numeric_cols}"
            )

        # تمیز کردن target_column ارسالی برای مقایسه
        target_column_cleaned = request.target_column.strip()
        if target_column_cleaned not in data_processor.df.columns:
            logging.error(f"ستون هدف {target_column_cleaned} در دیتاست وجود ندارد: {data_processor.df.columns.tolist()}")
            raise HTTPException(
                status_code=400,
                detail=f"ستون هدف {target_column_cleaned} در دیتاست وجود ندارد. ستون‌های موجود: {data_processor.df.columns.tolist()}"
            )

        if target_column_cleaned not in numeric_cols:
            logging.error(f"ستون هدف {target_column_cleaned} غیرعددی است: {numeric_cols}")
            raise HTTPException(
                status_code=400,
                detail=f"ستون هدف {target_column_cleaned} باید عددی باشد. ستون‌های عددی موجود: {numeric_cols}"
            )

        response_data = await predictor.train_and_predict(request.chat_id, target_column_cleaned)
        return response_data
    except Exception as e:
        logging.error(f"خطا در پیش‌بینی با ستون هدف: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_results/{chat_id}")
async def get_results(chat_id: int):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT result_json FROM dbo.result_table WHERE chat_id = ?", (chat_id,))
        results = cursor.fetchall()
        if not results:
            logging.error(f"هیچ نتیجه‌ای برای chat_id {chat_id} یافت نشد.")
            raise HTTPException(status_code=404, detail=f"هیچ نتیجه‌ای برای chat_id {chat_id} یافت نشد.")

        result_list = [json.loads(row[0]) for row in results]
        logging.debug(f"نتایج برای chat_id {chat_id} بازیابی شدند.")
        return {"chat_id": chat_id, "results": result_list}
    except Exception as e:
        logging.error(f"خطا در بازیابی نتایج برای chat_id {chat_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)