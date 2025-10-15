import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os
from sklearn.impute import KNNImputer
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import arabic_reshaper
from bidi.algorithm import get_display
import json

# Access the API key
api_key = "AIzaSyB8Rz8vHUO0ASP90_QF7VR9pvkXYWgfH_I"  # کلید API خود را اینجا قرار دهید
if not api_key:
    raise ValueError("GEMINI_API_KEY not found or is empty")

try:
    import xgboost as xgb
except ImportError:
    xgb = None

app = FastAPI(title="Advanced Financial Predictor API")

class DataProcessor:
    def __init__(self):
        self.df = None

    async def load_file(self, file: UploadFile):
        try:
            if file.filename.endswith('.csv'):
                self.df = pd.read_csv(file.file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file.file)
            else:
                raise ValueError("فرمت فایل پشتیبانی نمی‌شود. فقط فایل‌های CSV و Excel مجاز هستند.")
            
            if self.df.empty:
                raise ValueError("فایل بارگذاری‌شده خالی است.")
            
            clean_report = self.clean_data()
            mine_report = self.mine_data()
            
            return {
                "message": "فایل با موفقیت بارگذاری شد!",
                "columns": self.df.columns.tolist(),
                "cleaning_report": clean_report,
                "mining_report": mine_report
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطا در بارگذاری فایل: {str(e)}")

    def clean_data(self):
        if self.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")

        df_cleaned = self.df.copy()
        columns_before = df_cleaned.columns.tolist()
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        columns_after = df_cleaned.columns.tolist()
        dropped_columns = set(columns_before) - set(columns_after)

        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns

        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=5)
            df_cleaned[numeric_cols] = pd.DataFrame(imputer.fit_transform(df_cleaned[numeric_cols]), columns=numeric_cols)

        if not non_numeric_cols.empty:
            for col in non_numeric_cols:
                mode_value = df_cleaned[col].mode()
                df_cleaned[col] = df_cleaned[col].fillna(mode_value[0] if not mode_value.empty else '')

        for col in non_numeric_cols:
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            except:
                pass

        initial_rows = len(df_cleaned)
        for col in numeric_cols:
            if col in df_cleaned.columns and df_cleaned[col].var() > 0:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                if not np.isnan(IQR) and not np.isinf(IQR):
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        if len(df_cleaned) < 2 or df_cleaned[numeric_cols].dropna().empty:
            raise HTTPException(status_code=400, detail="داده‌های کافی یا ستون‌های عددی معتبر باقی نمانده است.")

        self.df = df_cleaned
        return {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_cleaned),
            "numeric_columns": numeric_cols.tolist(),
            "non_numeric_columns": non_numeric_cols.tolist(),
            "dropped_columns": list(dropped_columns),
            "message": "مقادیر گمشده پر شدند و داده‌های پرت حذف شدند."
        }

    def mine_data(self):
        if self.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")

        desc_stats = self.df.describe(include='all').replace([np.inf, -np.inf], np.nan).fillna(0).to_dict()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0).to_dict() if not numeric_cols.empty else {}

        outlier_report = {}
        for col in numeric_cols:
            if self.df[col].var() > 0:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if np.isnan(IQR) or np.isinf(IQR):
                    outlier_report[col] = 0
                else:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                    outlier_report[col] = len(outliers)
            else:
                outlier_report[col] = 0

        return {
            "descriptive_stats": desc_stats,
            "correlation_matrix": corr_matrix,
            "outlier_report": outlier_report
        }

class Predictor:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        try:
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.5-pro"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطا در اتصال به Gemini API: {str(e)}")

    def analyze_dataset_with_gemini(self, df):
        try:
            sample_size = min(100, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

            numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            all_cols = df.columns.tolist()
            desc_stats = df_sample.describe(include='all').to_string()
            corr_matrix = df_sample[numeric_cols].corr().to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
            num_rows, num_cols = df.shape
            missing_values = df.isnull().sum().sum()

            available_models = ["Linear Regression", "Random Forest", "Decision Tree", 
                               "Gradient Boosting", "SVR"]
            if xgb:
                available_models.append("XGBoost")

            prompt = f"""
            شما یک متخصص یادگیری ماشین هستید. من یک نمونه از دیتاست با مشخصات زیر دارم (نمونه شامل {sample_size} ردیف است):
            - تعداد ردیف‌های کل دیتاست: {num_rows}
            - تعداد ستون‌ها: {num_cols}
            - تمام ستون‌ها: {all_cols}
            - آمار توصیفی (بر اساس نمونه):
            {desc_stats}
            - ماتریس همبستگی (برای ستون‌های عددی نمونه):
            {corr_matrix}
            - تعداد مقادیر گمشده در کل دیتاست: {missing_values}

            با توجه به این اطلاعات:
            1. بهترین ستون برای استفاده به عنوان ستون هدف (target) در رگرسیون را پیشنهاد دهید. ستون هدف باید عددی باشد و بر اساس همبستگی، واریانس، یا اهمیت پیش‌بینی انتخاب شود.
            2. بهترین الگوریتم یادگیری ماشین برای رگرسیون را فقط از بین الگوریتم‌های زیر پیشنهاد دهید:
            {', '.join(available_models)}
            لطفاً فقط نام ستون هدف و نام الگوریتم را به صورت دقیق (مثلاً 'target_column: Sales' و 'model: Random Forest') و توضیح مختصری برای هر پیشنهاد ارائه دهید.
            """

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

            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text

            recommended_target = None
            recommended_model = None
            lower_response = response_text.lower()
            for col in numeric_cols:
                if col.lower() in lower_response and "target_column" in lower_response:
                    recommended_target = col
                    break
            
            for model_name in available_models:
                if model_name.lower() in lower_response:
                    recommended_model = model_name
                    break

            if recommended_target and recommended_model:
                return recommended_target, recommended_model, response_text
            else:
                return None, None, response_text

        except Exception as e:
            return None, None, f"خطا در تحلیل دیتاست با Gemini API: {str(e)}"

    async def train_and_predict(self):
        if self.data_processor.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
        try:
            recommended_target, recommended_model, recommendation_text = self.analyze_dataset_with_gemini(self.data_processor.df)
            if not recommended_target or not recommended_model:
                raise HTTPException(status_code=500, detail=f"Gemini نتوانست ستون هدف یا الگوریتم مناسبی پیشنهاد دهد: {recommendation_text}")

            target_column = recommended_target
            df_processed = pd.get_dummies(self.data_processor.df, drop_first=True)
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]

            if y.dtype not in [np.float64, np.float32, np.int64, np.int32]:
                raise HTTPException(status_code=400, detail="ستون هدف پیشنهادی باید عددی باشد.")

            if X.empty:
                raise HTTPException(status_code=400, detail="هیچ ستون برای ویژگی‌ها یافت نشد.")

            if len(X) < 2 or len(y) < 2:
                raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")

            X = X.loc[:, X.var(numeric_only=True) > 0]
            X = X.loc[:, X.notna().any()]
            X.fillna(X.mean(numeric_only=True), inplace=True)
            y.fillna(y.mean(), inplace=True)

            if X.empty or len(X.columns) == 0:
                raise HTTPException(status_code=400, detail="هیچ ویژگی معتبری برای آموزش مدل باقی نماند.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                raise HTTPException(status_code=400, detail="داده‌های استانداردشده شامل مقادیر نامعتبر (NaN یا بی‌نهایت) هستند.")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
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
                raise HTTPException(status_code=400, detail=f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")

            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                future_X = X_test[-5:]
                future_pred = model.predict(future_X)

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

                filename = f"prediction_{target_column}.png"
                with open(filename, "wb") as f:
                    f.write(buf.getvalue())

                response_data = {
                    "message": f"پیش‌بینی با مدل {recommended_model} انجام شد.",
                    "target_column": target_column,
                    "gemini_recommendation": recommendation_text,
                    "plot_url": f"/plot/{filename}",
                    "prediction_data": {
                        "actual_values": y_test.tolist(),
                        "predicted_values": y_pred.tolist(),
                        "future_predictions": future_pred.tolist()
                    }
                }

                output_filename = f"prediction_data_{target_column}.json"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=4)

                return JSONResponse(content=response_data)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"خطا در آموزش مدل {recommended_model}: {str(e)}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطا در فرآیند پیش‌بینی: {str(e)}")

data_processor = DataProcessor()
predictor = Predictor(data_processor)

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV یا Excel بارگذاری کنید.")
    return await data_processor.load_file(file)

@app.post("/predict")
async def predict():
    return await predictor.train_and_predict()

@app.post("/predict_with_user_target")
async def predict_with_user_target(target_column: str = Form(...)):
    """
    Endpoint جدید برای پیش‌بینی با ستون هدف مشخص‌شده توسط کاربر و الگوریتم پیشنهادی Gemini.
    کاربر نام ستون هدف را وارد می‌کند و Gemini بهترین الگوریتم را از بین الگوریتم‌های موجود انتخاب می‌کند.
    """
    if data_processor.df is None:
        raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")

    if target_column not in data_processor.df.columns:
        raise HTTPException(status_code=400, detail=f"ستون {target_column} در دیتاست وجود ندارد.")

    try:
        if data_processor.df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise HTTPException(status_code=400, detail="ستون هدف باید عددی باشد.")

        available_models = ["Linear Regression", "Random Forest", "Decision Tree", 
                           "Gradient Boosting", "SVR"]
        if xgb:
            available_models.append("XGBoost")

        try:
            sample_size = min(100, len(data_processor.df))
            df_sample = data_processor.df.sample(n=sample_size, random_state=42) if len(data_processor.df) > sample_size else data_processor.df

            numeric_cols = data_processor.df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
            all_cols = data_processor.df.columns.tolist()
            desc_stats = df_sample.describe(include='all').to_string()
            corr_matrix = df_sample[numeric_cols].corr().to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
            num_rows, num_cols = data_processor.df.shape
            missing_values = data_processor.df.isnull().sum().sum()

            prompt = f"""
            شما یک متخصص یادگیری ماشین هستید. من یک نمونه از دیتاست با مشخصات زیر دارم (نمونه شامل {sample_size} ردیف است):
            - تعداد ردیف‌های کل دیتاست: {num_rows}
            - تعداد ستون‌ها: {num_cols}
            - تمام ستون‌ها: {all_cols}
            - ستون هدف انتخاب‌شده توسط کاربر: {target_column}
            - آمار توصیفی (بر اساس نمونه):
            {desc_stats}
            - ماتریس همبستگی (برای ستون‌های عددی نمونه):
            {corr_matrix}
            - تعداد مقادیر گمشده در کل دیتاست: {missing_values}

            با توجه به این اطلاعات:
            بهترین الگوریتم یادگیری ماشین برای رگرسیون را فقط از بین الگوریتم‌های زیر پیشنهاد دهید:
            {', '.join(available_models)}
            لطفاً فقط نام الگوریتم را به صورت دقیق (مثلاً 'model: Random Forest') و توضیح مختصری برای پیشنهاد خود ارائه دهید.
            """

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

            response_text = ""
            for chunk in predictor.client.models.generate_content_stream(
                model=predictor.model,
                contents=contents,
                config=generate_content_config,
            ):
                response_text += chunk.text

            recommended_model = None
            lower_response = response_text.lower()
            for model_name in available_models:
                if model_name.lower() in lower_response and "model" in lower_response:
                    recommended_model = model_name
                    break

            if not recommended_model:
                raise HTTPException(status_code=500, detail=f"Gemini نتوانست الگوریتم مناسبی از بین {available_models} پیشنهاد دهد: {response_text}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطا در تحلیل دیتاست با Gemini API: {str(e)}")

        df_processed = pd.get_dummies(data_processor.df, drop_first=True)
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]

        if X.empty:
            raise HTTPException(status_code=400, detail="هیچ ستون برای ویژگی‌ها یافت نشد.")

        if len(X) < 2 or len(y) < 2:
            raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")

        X = X.loc[:, X.var(numeric_only=True) > 0]
        X = X.loc[:, X.notna().any()]
        X.fillna(X.mean(numeric_only=True), inplace=True)
        y.fillna(y.mean(), inplace=True)

        if X.empty or len(X.columns) == 0:
            raise HTTPException(status_code=400, detail="هیچ ویژگی معتبری برای آموزش مدل باقی نماند.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            raise HTTPException(status_code=400, detail="داده‌های استانداردشده شامل مقادیر نامعتبر (NaN یا بی‌نهایت) هستند.")

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if len(X_test) == 0 or len(y_test) == 0:
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
            raise HTTPException(status_code=400, detail=f"الگوریتم پیشنهادی Gemini ({recommended_model}) در دسترس نیست.")

        try:
            model = models[recommended_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            future_X = X_test[-5:]
            future_pred = model.predict(future_X)

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

            filename = f"prediction_user_target_{target_column}.png"
            with open(filename, "wb") as f:
                f.write(buf.getvalue())

            response_data = {
                "message": f"پیش‌بینی با مدل {recommended_model} برای ستون {target_column} انجام شد.",
                "target_column": target_column,
                "selected_model": recommended_model,
                "gemini_recommendation": response_text,
                "plot_url": f"/plot/{filename}",
                "prediction_data": {
                    "actual_values": y_test.tolist(),
                    "predicted_values": y_pred.tolist(),
                    "future_predictions": future_pred.tolist()
                }
            }

            output_filename = f"prediction_data_user_target_{target_column}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

            return JSONResponse(content=response_data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطا در آموزش مدل {recommended_model}: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در فرآیند پیش‌بینی: {str(e)}")