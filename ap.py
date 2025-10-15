import pandas as pd
import numpy as np
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
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import arabic_reshaper
from bidi.algorithm import get_display
import os
import re

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
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyB8Rz8vHUO0ASP90_QF7VR9pvkXYWgfH_I")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found or is empty")

try:
    import xgboost as xgb
except ImportError:
    xgb = None

font_path = "C:/path/to/Vazir.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Vazir"
else:
    plt.rcParams["font.family"] = "sans-serif"

class DataProcessor:
    def __init__(self):
        self.df = None

    async def load_file(self, file: UploadFile, filename: str):
        file_content = await file.read()
        file_like = io.BytesIO(file_content)
        if filename.endswith('.csv'):
            self.df = pd.read_csv(file_like)
        elif filename.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(file_like)
        else:
            raise HTTPException(status_code=400, detail="فرمت فایل پشتیبانی نمی‌شود. فقط CSV یا Excel مجاز است.")

        if self.df.empty:
            raise HTTPException(status_code=400, detail="فایل خالی است.")

        clean_report = self.clean_data()
        mine_report = self.mine_data()

        return {
            "message": "فایل با موفقیت بارگذاری شد!",
            "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "cleaning_report": clean_report,
            "mining_report": mine_report
        }

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
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-pro"

    async def analyze_dataset_with_gemini(self, df, target_column=None):
        sample_size = min(500, len(df))
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        all_cols = df.columns.tolist()
        desc_stats = df_sample.describe(include='all').replace([np.inf, -np.inf], np.nan).fillna(0).to_string()
        corr_matrix = df_sample[numeric_cols].corr().replace([np.inf, -np.inf], np.nan).fillna(0).to_string() if not numeric_cols.empty else "هیچ ستون عددی وجود ندارد"
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().sum()

        if target_column:
            if target_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"ستون هدف {target_column} در دیتاست وجود ندارد.")
            if df[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
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

            Recommend the best column to use as the target column for regression from the numeric columns: {numeric_cols.tolist()}.
            The target column must be numeric and selected based on correlation, variance, or predictive importance.
            Recommend the best machine learning algorithm for regression from the following options:
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
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text is not None:
                response_text += chunk.text

        if not response_text:
            recommended_target = target_column or next((col for col in df.columns if col in df.select_dtypes(include=[np.number]).columns), None)
            recommended_model = "Linear Regression"
            return recommended_target, recommended_model, "پاسخ Gemini خالی بود."

        recommended_target = target_column
        recommended_model = None
        available_models = ["Linear Regression", "Random Forest", "Decision Tree",
                           "Gradient Boosting", "SVR"]
        if xgb:
            available_models.append("XGBoost")

        model_match = re.search(r"model: ([\w\s]+)", response_text, re.IGNORECASE)
        recommended_model = model_match.group(1) if model_match and model_match.group(1) in available_models else "Linear Regression"

        return recommended_target, recommended_model, response_text

    async def train_and_predict(self, target_column: Optional[str] = None):
        if self.data_processor.df is None:
            raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")

        recommended_target, recommended_model, recommendation_text = await self.analyze_dataset_with_gemini(self.data_processor.df, target_column)
        if not recommended_target:
            raise HTTPException(status_code=400, detail="هیچ ستون عددی معتبری برای هدف یافت نشد.")

        target_column = recommended_target
        df_processed = pd.get_dummies(self.data_processor.df.drop(columns=[target_column], errors='ignore'), drop_first=True)
        df_processed[target_column] = self.data_processor.df[target_column]

        if target_column not in df_processed.columns:
            raise HTTPException(status_code=400, detail=f"ستون هدف {target_column} در داده‌های پردازش‌شده وجود ندارد.")

        if df_processed[target_column].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            raise HTTPException(status_code=400, detail="ستون هدف باید عددی باشد.")

        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]

        if X.empty:
            raise HTTPException(status_code=400, detail="هیچ ستون برای ویژگی‌ها یافت نشد.")

        if len(X) < 2 or len(y) < 2:
            raise HTTPException(status_code=400, detail="داده‌های کافی برای آموزش مدل وجود ندارد.")

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            X_numeric = X[numeric_cols]
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_numeric.columns
            vif_data["VIF"] = [variance_inflation_factor(X_numeric.fillna(0).values, i) for i in range(X_numeric.shape[1])]
            high_vif_cols = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
            if high_vif_cols:
                X_numeric = X_numeric.drop(columns=high_vif_cols)
            X = X_numeric
        else:
            raise HTTPException(status_code=400, detail="هیچ ستون عددی برای ویژگی‌ها یافت نشد.")

        X = X.loc[:, X.var(numeric_only=True) > 0]
        X = X.loc[:, X.notna().any()]
        X.fillna(X.mean(numeric_only=True), inplace=True)
        y.fillna(y.mean(), inplace=True)

        if X.empty or len(X.columns) == 0:
            raise HTTPException(status_code=400, detail="هیچ ویژگی معتبری برای آموزش مدل باقی نماند.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            raise HTTPException(status_code=400, detail="داده‌های استانداردشده شامل مقادیر نامعتبر هستند.")

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
            recommended_model = "Linear Regression"

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
        plot_data_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

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
            }
        }

        return response_data

data_processor = DataProcessor()
predictor = Predictor(data_processor)

class PredictRequest(BaseModel):
    target_column: str

@app.get("/")
async def index():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/upload_file")
async def upload_file(file: UploadFile):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV یا Excel بارگذاری کنید.")
    return await data_processor.load_file(file, file.filename)

@app.get("/get_numeric_columns")
async def get_numeric_columns():
    if data_processor.df is None:
        raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
    numeric_cols = data_processor.df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="هیچ ستون عددی در دیتاست یافت نشد.")
    return {"numeric_columns": numeric_cols}

@app.post("/predict")
async def predict():
    return await predictor.train_and_predict()

@app.post("/predict_with_target")
async def predict_with_target(request: PredictRequest):
    if data_processor.df is None:
        raise HTTPException(status_code=400, detail="لطفاً ابتدا فایل CSV یا Excel را بارگذاری کنید.")
    data_processor.df.columns = data_processor.df.columns.str.strip()
    numeric_cols = data_processor.df.select_dtypes(include=[np.number]).columns.str.strip().tolist()

    if not numeric_cols:
        raise HTTPException(status_code=400, detail="هیچ ستون عددی در دیتاست یافت نشد.")

    if not request.target_column:
        raise HTTPException(
            status_code=400,
            detail=f"لطفاً یک ستون هدف عددی ارائه کنید. ستون‌های عددی موجود: {numeric_cols}"
        )

    target_column_cleaned = request.target_column.strip()
    if target_column_cleaned not in data_processor.df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"ستون هدف {target_column_cleaned} در دیتاست وجود ندارد. ستون‌های موجود: {data_processor.df.columns.tolist()}"
        )

    if target_column_cleaned not in numeric_cols:
        raise HTTPException(
            status_code=400,
            detail=f"ستون هدف {target_column_cleaned} باید عددی باشد. ستون‌های عددی موجود: {numeric_cols}"
        )

    return await predictor.train_and_predict(target_column_cleaned)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)