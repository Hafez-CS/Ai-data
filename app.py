# app.py - نسخه نهایی با قابلیت انتخاب "حالت انتخاب هدف" و DeepSeek AI

import os
import io
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import httpx # 👈 جدید: برای فراخوانی API
from dotenv import load_dotenv # 👈 جدید: برای لود کردن متغیرهای محیطی

from fastapi import FastAPI, UploadFile, File, HTTPException, Form 
from fastapi.responses import JSONResponse
from pydantic import BaseModel 

from anyio import to_thread

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Algorithm Imports (XGBoost and LightGBM are optional)
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# Allowed Algorithms
ALLOWED_REGRESSORS = ["LinearRegression", "RandomForestRegressor"]
if XGBRegressor:
    ALLOWED_REGRESSORS.append("XGBRegressor")
if LGBMRegressor:
    ALLOWED_REGRESSORS.append("LGBMRegressor")


# ===================== تنظیمات محیطی و AI =====================

load_dotenv() # 👈 لود کردن متغیرهای محیطی از .env

AI_PROVIDER = os.getenv("AI_PROVIDER")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat")


# ===================== FastAPI Setup =====================

app = FastAPI(
    title="Smart Financial Forecasting Platform (Unified API)",
    description=("Supports Regression Models with AI Target Selection."),
    version="2.3.0", # نسخه به روز شده
)
PLOT_DIR = "static/plots"  
DATA_DIR = "static/data"    

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

from fastapi.staticfiles import StaticFiles
app.mount("/static/plots", StaticFiles(directory=PLOT_DIR), name="static_plots")
app.mount("/static/data", StaticFiles(directory=DATA_DIR), name="static_data")


class FullAnalysisRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    max_hist: int = 6
    max_rows_preview: int = 5


# ===================== توابع کمکی (بدون تغییر) =====================

def generate_plot(plot_func, input_filename: str, *args, target_name=None, title=None, xlabel=None, ylabel=None, custom_logic=None, **kwargs) -> str:
    """Generates and saves the matplotlib plot to PLOT_DIR."""
    plt.figure(figsize=(12, 7))
    plot_func(*args, **kwargs)
    
    if custom_logic:
        custom_logic() 

    if title: plt.title(title) 
    if xlabel: plt.xlabel(xlabel) 
    if ylabel: plt.ylabel(ylabel) 
    plt.grid(True)
    plt.tight_layout()
    
    base_name = os.path.splitext(input_filename)[0]
    safe_target_name = "".join(c for c in target_name if c.isalnum() or c in ('_', '-')).strip().lower() if target_name else "default"
    file_name = f"{base_name}_{safe_target_name}_chart.png"
    
    save_path = os.path.join(PLOT_DIR, file_name) 
    plt.savefig(save_path, format="png", dpi=150) 
    plt.close() 
    
    return file_name


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").copy()
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
                if coerced.notna().mean() > 0.5:
                    df[col] = coerced
            except Exception:
                pass
    return df

def split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() > 1]
    return num_cols, cat_cols

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = split_cols(X)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]) if cat_cols else "drop"
    return ColumnTransformer(transformers=[("num", num_pipe, num_cols),("cat", cat_pipe, cat_cols)],remainder="drop")

# تابع Fallback داخلی (در صورت عدم دسترسی به AI استفاده می‌شود)
def fallback_algo_and_target(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns.tolist()
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if "RandomForestRegressor" in ALLOWED_REGRESSORS: algo = "RandomForestRegressor"
    elif "XGBRegressor" in ALLOWED_REGRESSORS: algo = "XGBRegressor"
    else: algo = "LinearRegression"
    
    cand_names = ["profit","net_profit","revenue","income","sales","turnover","cost","expense","net_income", "charges", "price", "value", "ph", "quality"] 
    lower_cols = [c.lower() for c in cols]
    probable_target = None
    for nm in cand_names:
        if nm in lower_cols:
            # پیدا کردن ستون اصلی که نام کوچک‌شده آن تطابق دارد
            probable_target = cols[lower_cols.index(nm)] 
            break
    
    # اطمینان از اینکه ستون انتخاب شده عددی است
    if probable_target is None or probable_target not in num_cols: 
        probable_target = num_cols[-1] if num_cols else None
    
    return {"algorithm": algo, "target_column": probable_target}

# ===================== تابع فراخوانی DeepSeek AI 👈 جدید =====================

async def get_ai_suggestions_from_deepseek(df: pd.DataFrame, allowed_regressors: List[str]) -> Dict[str, Any]:
    """
    از DeepSeek API برای انتخاب هوشمندانه بهترین ستون هدف و الگوریتم رگرسیون استفاده می‌کند.
    در صورت خطا، به تابع fallback داخلی برمی‌گردد.
    """
    if AI_PROVIDER != "deepseek" or not DEEPSEEK_API_KEY:
        print("DeepSeek API not configured or API key missing. Falling back to internal logic.")
        return fallback_algo_and_target(df) 
    
    # تهیه خلاصه اطلاعات برای DeepSeek
    # نمایش یک سطر از دیتافریم (نمونه‌ای از نوع داده‌ها)
    column_info = df.head(1).T.to_dict().get(0, {}) 
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    prompt_message = f"""
    You are an AI expert in financial data analysis and machine learning model selection.
    Your task is to select the BEST target column (for regression) and a suitable algorithm 
    from the allowed list for a predictive model based on the provided CSV data.
    
    1. The target column MUST be **numeric** and present in the 'Available Numeric Columns for Target' list.
    2. The selected algorithm MUST be from the 'Allowed Algorithms' list.
    3. Prioritize columns related to financial outcomes (e.g., 'profit', 'revenue', 'price', 'value') 
       or measurable outcomes ('quality', 'ph') as the target.
    
    Available Columns (Type/Sample Value - First Row):
    {column_info}
    
    Available Numeric Columns for Target:
    {numeric_cols}
    
    Allowed Algorithms (Select ONE):
    {allowed_regressors}
    
    Output MUST be a single JSON object (with no other text/markdown or explanation) like this:
    {{"target_column": "SelectedColumnName", "algorithm": "SelectedAlgorithmName"}}
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert ML model selector who outputs only a single JSON object."},
            {"role": "user", "content": prompt_message}
        ],
        "temperature": 0.1,
        "max_tokens": 150 # فضای کافی برای پاسخ JSON
    }

    try:
        async with httpx.AsyncClient(base_url=DEEPSEEK_BASE_URL) as client:
            print(f"Calling DeepSeek at {DEEPSEEK_BASE_URL}/chat/completions with model {DEEPSEEK_MODEL}")
            response = await client.post("/chat/completions", headers=headers, json=payload, timeout=30.0)
            response.raise_for_status() 
            
            response_json = response.json()
            raw_content = response_json['choices'][0]['message']['content'].strip()

            # تلاش برای تمیز کردن خروجی (اگر در Markdown JSON محصور شده باشد)
            if raw_content.startswith("```json"):
                raw_content = raw_content.strip("```json").strip("```").strip()
                
            ai_choice = json.loads(raw_content)
            
            # اعتبارسنجی خروجی AI
            target = ai_choice.get("target_column")
            algo = ai_choice.get("algorithm")

            if target in df.columns and algo in allowed_regressors:
                print(f"DeepSeek selected target: {target}, algorithm: {algo}")
                return {"target_column": target, "algorithm": algo}
            else:
                print(f"DeepSeek output invalid or column/algo not found. Falling back. Target: {target}, Algo: {algo}")
                return fallback_algo_and_target(df)
            
    except Exception as e:
        print(f"Error calling DeepSeek API: {type(e).__name__}: {e}. Falling back to internal logic.")
        return fallback_algo_and_target(df)

# ===================== Prediction Logic (بدون تغییر) =====================
def run_prediction(df: pd.DataFrame, target: str, algo: str, input_filename: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Runs the ML model, generates files, and extracts prediction data."""
    
    print(f"--- START: Running prediction for target: {target} with {algo}. Data size: {df.shape}") 

    if target not in df.columns: raise ValueError(f"Target column '{target}' not found in data.")
    if not pd.api.types.is_numeric_dtype(df[target]): df[target] = pd.to_numeric(df[target], errors="coerce")
    clean_df = df.dropna(subset=[target])
    if clean_df.empty or clean_df.shape[0] < 5: raise ValueError("Data is too small or target column is empty after cleaning.")

    feature_cols = [c for c in clean_df.columns if c != target]
    if not feature_cols: raise ValueError("Data contains only the target column after cleaning. Cannot train a model.")
        
    X = clean_df[feature_cols] 
    y = clean_df[target]
    pre = make_preprocessor(X)
    
    # Model Selection
    if algo == "XGBRegressor" and XGBRegressor:
        model_instance = XGBRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    elif algo == "LGBMRegressor" and LGBMRegressor:
        model_instance = LGBMRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    elif algo == "RandomForestRegressor":
        model_instance = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    elif algo == "LinearRegression":
        model_instance = LinearRegression()
    else:
        # Fallback to default if selected algo is unavailable
        algo = "LinearRegression"
        model_instance = LinearRegression()


    pipe = Pipeline([("preprocess", pre), ("model", model_instance)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    
    pipe.fit(X_train, y_train) 
    print("--- MID: Model training completed.") 

    # --- Metrics Calculation ---
    if X_test.empty:
        y_pred_test = np.array([])
        mae = float('nan') 
        rmse = float('nan') 
    else:
        y_pred_test = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    # --- End Metrics Calculation ---

    # Future Prediction
    FUTURE_STEPS = 5 
    if X_test.shape[0] > 0:
        last_X_test_row = X_test.iloc[[X_test.shape[0] - 1]]
        X_future = pd.concat([last_X_test_row] * FUTURE_STEPS, ignore_index=True)
        y_future_pred = pipe.predict(X_future)
    else:
        y_future_pred = np.array([])
    y_future_pred_list = y_future_pred.tolist()
    
    # Combine raw data
    y_actual_full = y_train.tolist() + y_test.tolist()
    y_pred_train = pipe.predict(X_train)
    y_pred_full = y_pred_train.tolist() + y_pred_test.tolist()
    
    # Apply Smoothing (Moving Average)
    SMOOTHING_WINDOW = max(5, int(len(y_actual_full) * 0.01)) 
    y_actual_series = pd.Series(y_actual_full)
    y_pred_series = pd.Series(y_pred_full)
    y_actual_smooth = y_actual_series.rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean().tolist()
    y_pred_smooth = y_pred_series.rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean().tolist()

    # Plotting Logic (uses SMOOTHED data)
    indices_full = np.arange(len(y_actual_full)) 
    indices_future = np.arange(len(y_actual_full) - 1, len(y_actual_full) + FUTURE_STEPS) 
    
    def add_future_line():
        future_plot_data = [y_pred_smooth[-1]] + y_future_pred_list
        plt.plot(indices_full, y_actual_smooth, label='Actual Values (Smoothed)', color='blue', linewidth=2) 
        plt.plot(indices_full, y_pred_smooth, label='Predicted Values (Smoothed)', color='orange', linewidth=2)
        plt.plot(indices_future, future_plot_data, label='Future Prediction', color='green', linestyle='--', linewidth=2)
        plt.legend() 

    # ارسال target_name به generate_plot برای نام‌گذاری خروجی
    chart_filename = generate_plot(
        plt.plot, input_filename,
        [0], [0], target_name=target,
        title=f"Smoothed Prediction of {target} using {algo} model (Window={SMOOTHING_WINDOW})", xlabel="Data Index", ylabel=target,
        custom_logic=add_future_line
    )
    print(f"--- MID: Plot generated: {chart_filename}") 
    
    # ذخیره داده‌های JSON
    base_name = os.path.splitext(input_filename)[0]
    safe_target_name = "".join(c for c in target if c.isalnum() or c in ('_', '-')).strip().lower()
    data_filename = f"{base_name}_{safe_target_name}_data.json"
    data_save_path = os.path.join(DATA_DIR, data_filename) 
    
    raw_prediction_data = {
        "actual_values": y_actual_smooth, 
        "predicted_values": y_pred_smooth,
        "future_predictions": y_future_pred_list,
        "smoothing_window": SMOOTHING_WINDOW
    }
    
    with open(data_save_path, 'w') as f:
        json.dump(raw_prediction_data, f, indent=4)
        
    print("--- END: Prediction function finished successfully.")
        
    return {
        "algorithm_used": algo,
        "target_used": target,
        "metrics": {"MAE": mae, "RMSE": rmse},
        "prediction_chart_filename": chart_filename,
        "prediction_data_url": f"/static/data/{data_filename}", 
        "prediction_status": "Success"
    }


# ===================== Unified Endpoint (POST) 👈 تغییرات اصلی در منطق AI =====================
@app.post("/full_analysis")
async def full_analysis(
    file: UploadFile = File(..., description="The CSV file containing financial data."),
    config: str = Form(
        '{"test_size":0.2,"random_state":42,"max_hist":6,"max_rows_preview":5}',
        description="Configuration JSON as a string (e.g., '{\"test_size\":0.2, ...}')."
    ),
    selection_mode: str = Form(..., description="Selection mode: 'ai' for automatic selection or 'user' for manual."),
    target_column: Optional[str] = Form(None, description="The specific target column name if selection_mode is 'user'.")
):
    
    all_columns = []
    target_to_use = "N/A" 
    algo_to_use = "RandomForestRegressor" # پیش‌فرض داخلی
    
    # بررسی صحت selection_mode
    if selection_mode not in ["ai", "user"]:
        raise HTTPException(status_code=400, detail="selection_mode must be 'ai' or 'user'.")
        
    try:
        config_dict = json.loads(config)
        config_model = FullAnalysisRequest(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config format: {e}")
        
    try:
        input_filename = file.filename
        if not input_filename: input_filename = "uploaded_data.csv"
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content)) 
        df = basic_clean(df)
        
        all_columns = df.columns.tolist() 

        # 1. منطق تعیین ستون هدف
        if selection_mode == "user":
            # حالت کاربر: ستون هدف باید توسط کاربر ارسال شده باشد
            if not target_column or target_column not in all_columns:
                return JSONResponse(status_code=400, content={
                    "message": "در حالت انتخابی کاربر، ستون هدف معتبر و موجود در فایل باید ارسال شود.",
                    "all_columns": all_columns,
                    "target_column": target_column if target_column else "N/A",
                    "initial_status": "Error"
                })
            target_to_use = target_column
            
        else: # selection_mode == "ai"
            # 💥 فراخوانی تابع AI
            ai_suggestions = await get_ai_suggestions_from_deepseek(df, ALLOWED_REGRESSORS)
            
            target_to_use = ai_suggestions.get("target_column")
            algo_to_use = ai_suggestions.get("algorithm", algo_to_use)
            
            if target_to_use is None or target_to_use not in all_columns: 
                 # اگر AI نتوانست ستون معتبری پیدا کند
                return JSONResponse(content={
                     "message": "AI نتوانست ستون هدف معتبری پیدا کند. لطفاً خودتان از لیست زیر یک ستون انتخاب کنید.",
                     "all_columns": all_columns,
                     "target_column": "N/A",
                     "initial_status": "ColumnSelectionRequired"
                 })

        # 2. اجرای پیش‌بینی (استفاده از to_thread برای تابع CPU-Bound)
        prediction_results = await to_thread.run_sync(
            run_prediction, 
            df, target_to_use, algo_to_use, input_filename, config_model.test_size, config_model.random_state
        )

        # 3. تولید خروجی نهایی
        final_response = {
            "message": f"تحلیل مالی با ستون '{target_to_use}' انجام شد.",
            "target_column": prediction_results.get("target_used", "N/A"),
            "algorithm_used": prediction_results.get("algorithm_used", "N/A"),
            "plot_url": f"/static/plots/{prediction_results.get('prediction_chart_filename', '')}", 
            "prediction_data_url": prediction_results.get('prediction_data_url', ''), 
            "metrics": prediction_results.get("metrics", {}),
            "all_columns": all_columns, 
            "initial_status": "Success" 
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        # مدیریت خطاهای داخلی - اطمینان از وجود all_columns
        error_message = f"خطا در اجرای مدل: {type(e).__name__}: {str(e)}"
        return JSONResponse(
            status_code=500, 
            content={
                "message": error_message,
                "all_columns": all_columns,
                "target_column": target_to_use,
                "initial_status": "Error"
            }
        )

    return JSONResponse(content=final_response)
