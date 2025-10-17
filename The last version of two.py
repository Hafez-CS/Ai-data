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
from google import genai
from google.genai import types
import arabic_reshaper
from bidi.algorithm import get_display
import json

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Access the API key
api_key = "AIzaSyB8Rz8vHUO0ASP90_QF7VR9pvkXYWgfH_I"
if not api_key:
    raise ValueError("GEMINI_API_KEY not found or is empty")

def financial_predictor(file_path: str, target_column: Optional[str] = None):
    """
    Process a financial dataset and perform regression predictions.
    
    Args:
        file_path (str): Path to the CSV or Excel file
        target_column (str, optional): Target column for prediction. If None, Gemini will select one.
    
    Returns:
        dict: Prediction results including message, target column, model used, Gemini recommendation,
              plot filename, and prediction data
    """
    class DataProcessor:
        def __init__(self):
            self.df = None

        def load_file(self, file_path):
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    raise ValueError("File format not supported. Only CSV and Excel files are allowed.")
                
                if self.df.empty:
                    raise ValueError("The uploaded file is empty.")
                
                return self.clean_data()
            except Exception as e:
                raise ValueError(f"Error loading file: {str(e)}")

        def clean_data(self):
            if self.df is None:
                raise ValueError("Please load a CSV or Excel file first.")

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
                raise ValueError("Insufficient data or no valid numeric columns remaining.")

            self.df = df_cleaned
            return {
                "initial_rows": initial_rows,
                "cleaned_rows": len(df_cleaned),
                "numeric_columns": numeric_cols.tolist(),
                "non_numeric_columns": non_numeric_cols.tolist(),
                "dropped_columns": list(dropped_columns),
                "message": "Missing values filled and outliers removed."
            }

    class Predictor:
        def __init__(self, data_processor):
            self.data_processor = data_processor
            try:
                self.client = genai.Client(api_key=api_key)
                self.model = "gemini-2.5-pro"
            except Exception as e:
                raise ValueError(f"Error connecting to Gemini API: {str(e)}")

        def analyze_dataset_with_gemini(self, df, user_target_column=None):
            try:
                sample_size = min(100, len(df))
                df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

                numeric_cols = df.select_dtypes(include=[np.float64, np.float32, np.int64, np.int32]).columns
                all_cols = df.columns.tolist()
                desc_stats = df_sample.describe(include='all').to_string()
                corr_matrix = df_sample[numeric_cols].corr().to_string() if not numeric_cols.empty else "No numeric columns available"
                num_rows, num_cols = df.shape
                missing_values = df.isnull().sum().sum()

                available_models = ["Linear Regression", "Random Forest", "Decision Tree", 
                                 "Gradient Boosting", "SVR"]
                if xgb:
                    available_models.append("XGBoost")

                if user_target_column:
                    prompt = f"""
                    You are a machine learning expert. I have a dataset sample with the following details (sample size: {sample_size} rows):
                    - Total rows in dataset: {num_rows}
                    - Number of columns: {num_cols}
                    - All columns: {all_cols}
                    - User-selected target column: {user_target_column}
                    - Descriptive statistics (based on sample):
                    {desc_stats}
                    - Correlation matrix (for numeric columns in sample):
                    {corr_matrix}
                    - Total missing values in dataset: {missing_values}

                    Based on this information:
                    Recommend the best machine learning algorithm for regression from the following:
                    {', '.join(available_models)}
                    Provide only the algorithm name (e.g., 'model: Random Forest') and a brief explanation for your recommendation.
                    """
                else:
                    prompt = f"""
                    You are a machine learning expert. I have a dataset sample with the following details (sample size: {sample_size} rows):
                    - Total rows in dataset: {num_rows}
                    - Number of columns: {num_cols}
                    - All columns: {all_cols}
                    - Descriptive statistics (based on sample):
                    {desc_stats}
                    - Correlation matrix (for numeric columns in sample):
                    {corr_matrix}
                    - Total missing values in dataset: {missing_values}

                    Based on this information:
                    1. Recommend the best target column for regression (must be numeric, chosen based on correlation, variance, or predictive importance).
                    2. Recommend the best machine learning algorithm for regression from the following:
                    {', '.join(available_models)}
                    Provide only the target column name (e.g., 'target_column: Sales'), the algorithm name (e.g., 'model: Random Forest'), and a brief explanation for each recommendation.
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
                    response_text += chunk.text

                recommended_target = user_target_column
                recommended_model = None
                lower_response = response_text.lower()
                
                if not user_target_column:
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
                return None, None, f"Error analyzing dataset with Gemini API: {str(e)}"

        def train_and_predict(self, user_target_column=None):
            if self.data_processor.df is None:
                raise ValueError("Please load a CSV or Excel file first.")

            recommended_target, recommended_model, recommendation_text = self.analyze_dataset_with_gemini(
                self.data_processor.df, user_target_column
            )
            if not recommended_target or not recommended_model:
                raise ValueError(f"Gemini could not recommend a suitable target or model: {recommendation_text}")

            target_column = recommended_target
            df_processed = pd.get_dummies(self.data_processor.df, drop_first=True)
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]

            if y.dtype not in [np.float64, np.float32, np.int64, np.int32]:
                raise ValueError("The target column must be numeric.")

            if X.empty:
                raise ValueError("No columns found for features.")

            if len(X) < 2 or len(y) < 2:
                raise ValueError("Insufficient data to train the model.")

            X = X.loc[:, X.var(numeric_only=True) > 0]
            X = X.loc[:, X.notna().any()]
            X.fillna(X.mean(numeric_only=True), inplace=True)
            y.fillna(y.mean(), inplace=True)

            if X.empty or len(X.columns) == 0:
                raise ValueError("No valid features remaining for model training.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                raise ValueError("Standardized data contains invalid values (NaN or infinite).")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            if len(X_test) == 0 or len(y_test) == 0:
                raise ValueError("Insufficient test data.")

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
                raise ValueError(f"Recommended model {recommended_model} is not available.")

            try:
                model = models[recommended_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                future_X = X_test[-5:]
                future_pred = model.predict(future_X)

                fig = plt.Figure(figsize=(14, 8))
                ax = fig.add_subplot(111)
                indices = np.arange(len(y_test))

                ax.plot(indices, y_test.values, color='blue', label=get_display(arabic_reshaper.reshape('Actual Values')), linewidth=2)
                ax.plot(indices, y_pred, color='orange', label=get_display(arabic_reshaper.reshape('Predicted Values')), linewidth=2)
                ax.plot(np.arange(len(y_test), len(y_test) + 5), future_pred, color='green', linestyle='--',
                        label=get_display(arabic_reshaper.reshape('Future Predictions')), linewidth=2)

                ax.set_xlabel(get_display(arabic_reshaper.reshape("Data Index")))
                ax.set_ylabel(get_display(arabic_reshaper.reshape("Values")))
                ax.set_title(get_display(arabic_reshaper.reshape(f"Prediction of {target_column} using {recommended_model}")))
                ax.legend()
                ax.grid(True)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

                canvas = FigureCanvas(fig)
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)

                filename = f"prediction_{'user_target_' if user_target_column else ''}{target_column}.png"
                with open(filename, "wb") as f:
                    f.write(buf.getvalue())

                response_data = {
                    "message": f"Prediction performed using {recommended_model} for column {target_column}.",
                    "target_column": target_column,
                    "selected_model": recommended_model,
                    "gemini_recommendation": recommendation_text,
                    "plot_filename": filename,
                    "prediction_data": {
                        "actual_values": y_test.tolist(),
                        "predicted_values": y_pred.tolist(),
                        "future_predictions": future_pred.tolist()
                    }
                }

                output_filename = f"prediction_data_{'user_target_' if user_target_column else ''}{target_column}.json"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=4)

                return response_data

            except Exception as e:
                raise ValueError(f"Error training model {recommended_model}: {str(e)}")

    try:
        data_processor = DataProcessor()
        data_processor.load_file(file_path)
        predictor = Predictor(data_processor)
        return predictor.train_and_predict(target_column)
    except Exception as e:
        return {"error": f"Error in prediction process: {str(e)}"}

if __name__ == "__main__":
    # Example usage
    result = financial_predictor("path_to_your_file.csv", "your_target_column")
    print(result)