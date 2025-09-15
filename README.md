# پیش‌بینی مالی پیشرفته (Advanced Financial Predictor)

## معرفی
این پروژه شامل دو بخش اصلی است:
1. **برنامه گرافیکی (GUI)**: یک اپلیکیشن مبتنی بر PyQt5 برای تحلیل داده‌های مالی و پیش‌بینی مقادیر عددی با استفاده از مدل‌های یادگیری ماشین.
2. **API مبتنی بر FastAPI**: یک رابط برنامه‌نویسی کاربردی (API) برای بارگذاری داده‌ها، پاک‌سازی، داده‌کاوی، تولید نمودار، و پیش‌بینی خودکار با استفاده از مدل پیشنهادی Gemini.

هر دو بخش امکان بارگذاری فایل‌های CSV، پاک‌سازی داده‌ها، داده‌کاوی، مقایسه ستون‌ها، و پیش‌بینی خودکار را فراهم می‌کنند. نتایج پیش‌بینی در API به‌صورت یک گراف خطی آینده‌نگرانه نمایش داده می‌شود که برای کاربران غیرحرفه‌ای قابل فهم است. خطاها در فایل `app_errors.log` ثبت می‌شوند تا از کرش برنامه جلوگیری شود.

---

## Introduction
This project consists of two main components:
1. **Graphical Application (GUI)**: A PyQt5-based application for financial data analysis and prediction using machine learning models.
2. **FastAPI-based API**: An API for loading data, cleaning, mining, generating visualizations, and performing automatic predictions using the Gemini-recommended model.

Both components allow loading CSV files, data cleaning, data mining, column comparison, and automatic prediction. Prediction results in the API are displayed as a future-oriented line graph, designed to be intuitive for non-expert users. Errors are logged in `app_errors.log` to prevent crashes.

## ویژگی‌ها
### برنامه گرافیکی (GUI)
- **بارگذاری داده‌ها**: بارگذاری فایل‌های CSV برای تحلیل.
- **پاک‌سازی داده‌ها**: پر کردن مقادیر گمشده (میانگین برای ستون‌های عددی، مد برای ستون‌های غیرعددی) و حذف داده‌های پرت با روش IQR.
- **داده‌کاوی**: ارائه آمار توصیفی، ماتریس همبستگی، و گزارش داده‌های پرت.
- **مقایسه ستون‌ها**: نمایش نمودار پراکندگی دوبعدی یا سه‌بعدی برای ۲ یا ۳ ستون انتخابی (با Ctrl+کلیک).
- **نمایش پراکندگی داده‌ها**: نمایش نمودارهای جفتی (Pairplot) برای حداکثر ۴ ستون عددی.
- **پیش‌بینی خودکار**: انتخاب خودکار بهترین مدل یادگیری ماشین (بر اساس R²) از میان:
  - رگرسیون خطی (Linear Regression)
  - جنگل تصادفی (Random Forest Regressor)
  - درخت تصمیم (Decision Tree Regressor)
  - تقویت گرادیان (Gradient Boosting Regressor)
  - ماشین بردار پشتیبان (Support Vector Regressor)
  - XGBoost (در صورت نصب)
- **نمایش نتایج**: نمایش مقادیر واقعی و پیش‌بینی‌شده در جدول و نمودار پراکندگی.
- **مدیریت خطاها**: ثبت خطاها در فایل `app_errors.log` و نمایش پیام‌های خطا در نوار وضعیت.
- **رابط کاربری بهبودیافته**: استفاده از QGridLayout، نوار وضعیت، انتخاب چندگانه ستون‌ها، و استایل‌دهی مدرن.

### API (FastAPI)
- **بارگذاری داده‌ها**: آپلود فایل‌های CSV از طریق endpoint `/upload_csv`.
- **پاک‌سازی داده‌ها**: پاک‌سازی داده‌ها با پر کردن مقادیر گمشده و حذف داده‌های پرت از طریق endpoint `/clean_data`.
- **داده‌کاوی**: ارائه آمار توصیفی، ماتریس همبستگی، و گزارش داده‌های پرت از طریق endpoint `/mine_data`.
- **نمایش نمودارها**: تولید نمودارهای اولیه (پراکندگی، هیستوگرام، جعبه‌ای، و نقشه حرارتی همبستگی) از طریق endpoint `/visualize_data`.
- **مقایسه ستون‌ها**: نمایش نمودار پراکندگی دوبعدی یا سه‌بعدی برای ۲ یا ۳ ستون از طریق endpoint `/compare_columns`.
- **پیش‌بینی خودکار**: استفاده از مدل پیشنهادی Gemini برای پیش‌بینی و نمایش نتایج به‌صورت گراف خطی آینده‌نگرانه از طریق endpoint `/predict`.
  - مدل‌ها: Linear Regression، Random Forest، Decision Tree، Gradient Boosting، SVR، و XGBoost (در صورت نصب).
  - گراف خطی شامل مقادیر واقعی (خط آبی)، پیش‌بینی‌شده (خط نارنجی)، و پیش‌بینی آینده (خط سبز نقطه‌چین) برای ۵ نقطه فرضی.
- **مدیریت خطاها**: ثبت خطاها در `app_errors.log` و برگرداندن خطاها به‌صورت HTTPException.
- **پشتیبانی از فارسی**: استفاده از فونت Vazir برای نمایش صحیح متن فارسی در نمودارها.

---

## Features
### Graphical Application (GUI)
- **Data Loading**: Load CSV files for analysis.
- **Data Cleaning**: Fill missing values (mean for numeric, mode for non-numeric) and remove outliers using the IQR method.
- **Data Mining**: Provide descriptive statistics, correlation matrix, and outlier reports.
- **Column Comparison**: Display 2D or 3D scatter plots for 2 or 3 selected columns (using Ctrl+click).
- **Data Scatter Plot**: Show pairplots for up to 4 numeric columns.
- **Automatic Prediction**: Automatically select the best machine learning model (based on R²) from:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor
  - XGBoost (if installed)
- **Result Visualization**: Display actual vs. predicted values in a table and scatter plot.
- **Error Handling**: Log errors in `app_errors.log` and display error messages in the status bar.
- **Enhanced UI**: Uses QGridLayout, status bar, multi-column selection, and modern styling.

### API (FastAPI)
- **Data Loading**: Upload CSV files via the `/upload_csv` endpoint.
- **Data Cleaning**: Clean data by filling missing values and removing outliers via the `/clean_data` endpoint.
- **Data Mining**: Provide descriptive statistics, correlation matrix, and outlier reports via the `/mine_data` endpoint.
- **Visualizations**: Generate initial plots (scatter, histogram, box plot, and correlation heatmap) via the `/visualize_data` endpoint.
- **Column Comparison**: Display 2D or 3D scatter plots for 2 or 3 columns via the `/compare_columns` endpoint.
- **Automatic Prediction**: Use the Gemini-recommended model for prediction and display results as a future-oriented line graph via the `/predict` endpoint.
  - Models: Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, and XGBoost (if installed).
  - Line graph includes actual values (blue line), predicted values (orange line), and future predictions (green dashed line) for 5 hypothetical points.
- **Error Handling**: Log errors in `app_errors.log` and return errors as HTTPExceptions.
- **Persian Text Support**: Use the Vazir font for proper rendering of Persian text in plots.

## پیش‌نیازها
برای اجرای برنامه (GUI و API)، نیاز به نصب کتابخانه‌های زیر دارید:

```bash
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost arabic-reshaper python-bidi fastapi uvicorn google-genai
```

**توجه**:
- کتابخانه `xgboost` اختیاری است. اگر نصب نشود، گزینه XGBoost غیرفعال خواهد بود.
- برای API، نصب `fastapi` و `uvicorn` ضروری است.
- برای پیش‌بینی با Gemini، به یک API Key معتبر نیاز دارید.

### پشتیبانی از متن فارسی
برای نمایش صحیح متن فارسی در رابط کاربری و نمودارها:
1. فونت Vazir را از [اینجا](https://github.com/rastikerdar/vazir-font) دانلود کنید.
2. فونت را در سیستم نصب کنید:
   - **ویندوز**: فایل `.ttf` را به `C:\Windows\Fonts` کپی کنید.
   - **لینوکس/مک**: مراحل نصب فونت سیستم را دنبال کنید.
3. مسیر فونت را در فایل `app.py` (برای API) یا فایل‌های مرتبط با GUI تنظیم کنید:

```python
font_path = "path/to/Vazir.ttf"  # مسیر فایل فونت
```

### تنظیم API Key برای Gemini
برای استفاده از Gemini API:
1. یک API Key از Google دریافت کنید.
2. آن را به‌عنوان متغیر محیطی تنظیم کنید:

```bash
export GEMINI_API_KEY='your-api-key'
```

یا در کد به‌صورت پیش‌فرض تنظیم شده است (برای امنیت، توصیه می‌شود از متغیر محیطی استفاده کنید).

---

## Requirements
To run the application (GUI and API), install the following libraries:

```bash
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost arabic-reshaper python-bidi fastapi uvicorn google-genai
```

**Note**:
- The `xgboost` library is optional. If not installed, the XGBoost option will be unavailable.
- For the API, `fastapi` and `uvicorn` are required.
- For predictions with Gemini, a valid API Key is required.

### Persian Text Support
To render Persian text correctly in the GUI and plots:
1. Download the Vazir font from [here](https://github.com/rastikerdar/vazir-font).
2. Install the font on your system:
   - **Windows**: Copy the `.ttf` file to `C:\Windows\Fonts`.
   - **Linux/macOS**: Follow system-specific font installation steps.
3. Set the font path in `app.py` (for API) or GUI-related files:

```python
font_path = "path/to/Vazir.ttf"  # Path to the font file
```

### Setting Up Gemini API Key
To use the Gemini API:
1. Obtain an API Key from Google.
2. Set it as an environment variable:

```bash
export GEMINI_API_KEY='your-api-key'
```

Alternatively, a default key is set in the code (for security, use an environment variable).

## ساختار پروژه
پروژه به‌صورت ماژولار سازمان‌دهی شده و شامل فایل‌های زیر است:

### برنامه گرافیکی (GUI)
- `main.py`: نقطه ورود برنامه، راه‌اندازی اپلیکیشن PyQt5.
- `ui.py`: تنظیم رابط کاربری و ویجت‌ها.
- `data_processing.py`: منطق بارگذاری، پاک‌سازی، و داده‌کاوی.
- `visualization.py`: تولید نمودارهای اولیه، پراکندگی، و مقایسه.
- `prediction.py`: آموزش مدل پیشنهادی Gemini، پیش‌بینی، و نمایش گراف خطی آینده‌نگرانه.
- `dialog.py`: دیالوگ برای نمایش گزارش‌ها.
- `app_errors.log`: فایل لاگ برای ثبت خطاها.

### API (FastAPI)
- `app.py`: فایل اصلی API، شامل منطق بارگذاری، پاک‌سازی، داده‌کاوی، تولید نمودار، و پیش‌بینی با مدل پیشنهادی Gemini.
- `app_errors.log`: فایل لاگ برای ثبت خطاها.

---

## Project Structure
The project is organized modularly and includes the following files:

### Graphical Application (GUI)
- `main.py`: Entry point, initializes the PyQt5 application.
- `ui.py`: Sets up the user interface and widgets.
- `data_processing.py`: Handles data loading, cleaning, and mining.
- `visualization.py`: Generates initial, scatter, and comparison plots.
- `prediction.py`: Manages model selection with Gemini, prediction, and display of a future-oriented line graph.
- `dialog.py`: Dialog for displaying reports.
- `app_errors.log`: Log file for recording errors.

### API (FastAPI)
- `app.py`: Main API file, handling data loading, cleaning, mining, visualization, and prediction with the Gemini-recommended model.
- `app_errors.log`: Log file for recording errors.

## توابع پروژه
در این بخش، تمام توابع موجود در فایل‌های پروژه و عملکرد آن‌ها شرح داده شده است.

### **فایل `main.py` (GUI)**
- **تابع `__main__`**:
  - **عملکرد**: نقطه ورود برنامه. یک نمونه از `QApplication` و کلاس `AdvancedFinancialPredictorUI` ایجاد می‌کند، پنجره اصلی را نمایش می‌دهد و برنامه را اجرا می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به هیچ بخش خاصی از رابط کاربری متصل نیست؛ با اجرای `python main.py` فراخوانی می‌شود.

### **فایل `ui.py` (GUI)**
- **تابع `__init__`** (در کلاس `AdvancedFinancialPredictorUI`):
  - **عملکرد**: رابط کاربری گرافیکی را تنظیم می‌کند، شامل دکمه‌ها، منوی کشویی، لیست مقایسه (با قابلیت انتخاب چندگانه)، جدول نتایج، ناحیه نمودار، و نوار وضعیت. سیگنال‌های دکمه‌ها را به توابع مربوطه متصل می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `AdvancedFinancialPredictorUI` در `main.py` فراخوانی می‌شود.

### **فایل `data_processing.py` (GUI)**
کلاس `DataProcessor` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: نمونه‌ای از کلاس را ایجاد کرده و شیء والد (`parent`) را ذخیره می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `DataProcessor` در `ui.py` فراخوانی می‌شود.
- **تابع `load_csv`**:
  - **عملکرد**: یک فایل CSV را بارگذاری می‌کند، ستون‌های آن را در منوی کشویی هدف و لیست مقایسه نمایش می‌دهد و نمودارهای اولیه را تولید می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "بارگذاری فایل CSV" (`btn_load`) متصل است.
- **تابع `clean_data`**:
  - **عملکرد**: داده‌ها را پاک‌سازی می‌کند (پر کردن مقادیر گمشده با میانگین یا مد، حذف داده‌های پرت با روش IQR) و گزارش پاک‌سازی را نمایش می‌دهد. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "پاک‌سازی داده‌ها" (`btn_clean`) متصل است.
- **تابع `mine_data`**:
  - **عملکرد**: داده‌کاوی انجام می‌دهد، شامل آمار توصیفی، ماتریس همبستگی و گزارش داده‌های پرت، و نتایج را در دیالوگ نمایش می‌دهد. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "داده‌کاوی" (`btn_mine`) متصل است.

### **فایل `visualization.py` (GUI)**
کلاس `DataVisualizer` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: نمونه‌ای از کلاس را ایجاد کرده و شیء والد (`parent`) را ذخیره می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `DataVisualizer` در `ui.py` فراخوانی می‌شود.
- **تابع `visualize_data`**:
  - **عملکرد**: نمودارهای اولیه (پراکندگی، هیستوگرام، جعبه‌ای و نقشه حرارتی همبستگی) را برای ستون‌های عددی تولید می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: پس از بارگذاری فایل CSV (`load_csv`) یا پاک‌سازی داده‌ها (`clean_data`) فراخوانی می‌شود.
- **تابع `show_scatter_plot`**:
  - **عملکرد**: نمودارهای جفتی (Pairplot) را برای حداکثر ۴ ستون عددی تولید می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "نمودار پراکندگی" (`btn_scatter`) متصل است.
- **تابع `compare_columns`**:
  - **عملکرد**: نمودار پراکندگی دوبعدی (برای ۲ ستون) یا سه‌بعدی (برای ۳ ستون) را برای ستون‌های انتخاب‌شده تولید می‌کند. از انتخاب چندگانه (Ctrl+کلیک) پشتیبانی می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "مقایسه ستون‌ها" (`btn_compare`) متصل است.

### **فایل `prediction.py` (GUI)**
کلاس `Predictor` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: نمونه‌ای از کلاس را ایجاد کرده و شیء والد (`parent`) را ذخیره می‌کند. API Key برای Gemini را تنظیم می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `Predictor` در `ui.py` فراخوانی می‌شود.
- **تابع `analyze_dataset_with_gemini`**:
  - **عملکرد**: دیتاست را با Gemini API تحلیل می‌کند و بهترین مدل یادگیری ماشین را پیشنهاد می‌دهد. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: توسط تابع `train_and_predict` فراخوانی می‌شود.
- **تابع `train_and_predict`**:
  - **عملکرد**: داده‌ها را آماده می‌کند (استانداردسازی، تقسیم به مجموعه آموزش و آزمون)، مدل پیشنهادی Gemini را آموزش می‌دهد، پیش‌بینی انجام می‌دهد، و نتایج را به‌صورت گراف خطی آینده‌نگرانه (مقادیر واقعی، پیش‌بینی‌شده، و پیش‌بینی آینده) نمایش می‌دهد. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "شروع پیش‌بینی" (`btn_predict`) متصل است.

### **فایل `dialog.py` (GUI)**
کلاس `DataAnalysisDialog` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: یک دیالوگ برای نمایش گزارش‌ها (مثل گزارش پاک‌سازی یا داده‌کاوی) با کادر متنی و دکمه بستن ایجاد می‌کند.
  - **اتصال**: توسط `clean_data` و `mine_data` در `data_processing.py` فراخوانی می‌شود.

### **فایل `app.py` (API)**
- **کلاس `DataProcessor`**:
  - **تابع `__init__`**: نمونه‌ای از کلاس را ایجاد می‌کند.
  - **تابع `load_csv`**: فایل CSV را بارگذاری کرده و لیست ستون‌ها را برمی‌گرداند. متصل به endpoint `/upload_csv`.
  - **تابع `clean_data`**: داده‌ها را پاک‌سازی می‌کند (پر کردن مقادیر گمشده و حذف داده‌های پرت). متصل به endpoint `/clean_data`.
  - **تابع `mine_data`**: داده‌کاوی انجام می‌دهد (آمار توصیفی، ماتریس همبستگی، گزارش داده‌های پرت). متصل به endpoint `/mine_data`.
- **کلاس `DataVisualizer`**:
  - **تابع `__init__`**: نمونه‌ای از کلاس را ایجاد می‌کند.
  - **تابع `visualize_data`**: نمودارهای اولیه (پراکندگی، هیستوگرام، جعبه‌ای، نقشه حرارتی) را تولید می‌کند. متصل به endpoint `/visualize_data`.
  - **تابع `show_scatter_plot`**: نمودار پراکندگی برای ستون‌های انتخابی تولید می‌کند. متصل به endpoint `/scatter_plot`.
  - **تابع `compare_columns`**: نمودار پراکندگی دوبعدی یا سه‌بعدی برای ۲ یا ۳ ستون تولید می‌کند. متصل به endpoint `/compare_columns`.
- **کلاس `Predictor`**:
  - **تابع `__init__`**: نمونه‌ای از کلاس را ایجاد کرده و API Key برای Gemini را تنظیم می‌کند.
  - **تابع `analyze_dataset_with_gemini`**: دیتاست را با Gemini API تحلیل کرده و مدل پیشنهادی را برمی‌گرداند.
  - **تابع `train_and_predict`**: مدل پیشنهادی Gemini را آموزش می‌دهد، پیش‌بینی انجام می‌دهد، و گراف خطی آینده‌نگرانه را تولید می‌کند. متصل به endpoint `/predict`.
- **تابع `get_plot`**: فایل نمودار PNG را برمی‌گرداند. متصل به endpoint `/plot/{filename}`.
- **تابع `__main__`**: سرور FastAPI را اجرا می‌کند.

---

## Project Functions
This section lists all functions in the project files and their purposes.

### **File `main.py` (GUI)**
- **Function `__main__`**:
  - **Purpose**: Entry point. Creates a `QApplication` and `AdvancedFinancialPredictorUI` instance, displays the main window, and runs the application. Errors are logged in `app_errors.log`.
  - **Connection**: Invoked by running `python main.py`.

### **File `ui.py` (GUI)**
- **Function `__init__`** (in class `AdvancedFinancialPredictorUI`):
  - **Purpose**: Sets up the GUI, including buttons, dropdown, comparison list, results table, plot area, and status bar. Connects button signals to functions.
  - **Connection**: Called when creating an `AdvancedFinancialPredictorUI` instance in `main.py`.

### **File `data_processing.py` (GUI)**
The `DataProcessor` class includes:
- **Function `__init__`**:
  - **Purpose**: Initializes the class and stores the parent object.
  - **Connection**: Called when creating a `DataProcessor` instance in `ui.py`.
- **Function `load_csv`**:
  - **Purpose**: Loads a CSV file, populates the target dropdown and comparison list, and triggers initial visualizations. Errors are logged.
  - **Connection**: Connected to the "Load CSV File" button (`btn_load`).
- **Function `clean_data`**:
  - **Purpose**: Cleans data (fills missing values, removes outliers) and displays a report. Errors are logged.
  - **Connection**: Connected to the "Clean Data" button (`btn_clean`).
- **Function `mine_data`**:
  - **Purpose**: Performs data mining (descriptive stats, correlation, outlier report) and displays results. Errors are logged.
  - **Connection**: Connected to the "Data Mining" button (`btn_mine`).

### **File `visualization.py` (GUI)**
The `DataVisualizer` class includes:
- **Function `__init__`**:
  - **Purpose**: Initializes the class and stores the parent object.
  - **Connection**: Called when creating a `DataVisualizer` instance in `ui.py`.
- **Function `visualize_data`**:
  - **Purpose**: Generates initial plots (scatter, histogram, box plot, heatmap). Errors are logged.
  - **Connection**: Called after `load_csv` or `clean_data`.
- **Function `show_scatter_plot`**:
  - **Purpose**: Generates pairplots for up to 4 numeric columns. Errors are logged.
  - **Connection**: Connected to the "Show Scatter Plot" button (`btn_scatter`).
- **Function `compare_columns`**:
  - **Purpose**: Generates 2D or 3D scatter plots for 2 or 3 selected columns. Errors are logged.
  - **Connection**: Connected to the "Compare Columns" button (`btn_compare`).

### **File `prediction.py` (GUI)**
The `Predictor` class includes:
- **Function `__init__`**:
  - **Purpose**: Initializes the class, stores the parent object, and sets up the Gemini API Key.
  - **Connection**: Called when creating a `Predictor` instance in `ui.py`.
- **Function `analyze_dataset_with_gemini`**:
  - **Purpose**: Analyzes the dataset with Gemini API and returns the recommended model. Errors are logged.
  - **Connection**: Called by `train_and_predict`.
- **Function `train_and_predict`**:
  - **Purpose**: Prepares data, trains the Gemini-recommended model, performs predictions, and displays a future-oriented line graph. Errors are logged.
  - **Connection**: Connected to the "Start Prediction" button (`btn_predict`).

### **File `dialog.py` (GUI)**
The `DataAnalysisDialog` class includes:
- **Function `__init__`**:
  - **Purpose**: Creates a dialog for displaying reports (cleaning or mining) with a text box and close button.
  - **Connection**: Called by `clean_data` and `mine_data`.

### **File `app.py` (API)**
- **Class `DataProcessor`**:
  - **Function `__init__`**: Initializes the class.
  - **Function `load_csv`**: Loads a CSV file and returns column names. Connected to `/upload_csv`.
  - **Function `clean_data`**: Cleans data (fills missing values, removes outliers). Connected to `/clean_data`.
  - **Function `mine_data`**: Performs data mining (stats, correlation, outliers). Connected to `/mine_data`.
- **Class `DataVisualizer`**:
  - **Function `__init__`**: Initializes the class.
  - **Function `visualize_data`**: Generates initial plots (scatter, histogram, box plot, heatmap). Connected to `/visualize_data`.
  - **Function `show_scatter_plot`**: Generates scatter plots for selected columns. Connected to `/scatter_plot`.
  - **Function `compare_columns`**: Generates 2D or 3D scatter plots for 2 or 3 columns. Connected to `/compare_columns`.
- **Class `Predictor`**:
  - **Function `__init__`**: Initializes the class and sets up the Gemini API Key.
  - **Function `analyze_dataset_with_gemini`**: Analyzes the dataset with Gemini API.
  - **Function `train_and_predict`**: Trains the Gemini-recommended model and generates a future-oriented line graph. Connected to `/predict`.
- **Function `get_plot`**: Returns a PNG plot file. Connected to `/plot/{filename}`.
- **Function `__main__`**: Runs the FastAPI server.

## نصب و اجرا
### برنامه گرافیکی (GUI)
1. مخزن پروژه را کلون کنید یا فایل‌ها را در یک دایرکتوری ذخیره کنید.
2. کتابخانه‌های مورد نیاز را نصب کنید (بخش پیش‌نیازها).
3. برنامه را با اجرای فایل `main.py` اجرا کنید:

```bash
python main.py
```

### API (FastAPI)
1. کتابخانه‌های مورد نیاز را نصب کنید (بخش پیش‌نیازها).
2. API Key برای Gemini را تنظیم کنید (بخش تنظیم API Key).
3. سرور را با اجرای فایل `app.py` راه‌اندازی کنید:

```bash
python app.py
```

سرور روی `http://localhost:8000` اجرا می‌شود.

---

## Installation and Running
### Graphical Application (GUI)
1. Clone the repository or save the files in a directory.
2. Install the required libraries (see Requirements).
3. Run the application by executing `main.py`:

```bash
python main.py
```

### API (FastAPI)
1. Install the required libraries (see Requirements).
2. Set up the Gemini API Key (see Setting Up Gemini API Key).
3. Start the server by running `app.py`:

```bash
python app.py
```

The server runs on `http://localhost:8000`.

## راهنمای استفاده
### برنامه گرافیکی (GUI)
1. **بارگذاری فایل CSV**:
   - روی دکمه "بارگذاری فایل CSV" کلیک کنید و یک فایل CSV انتخاب کنید.
   - ستون‌های فایل در منوی کشویی هدف و لیست مقایسه ظاهر می‌شوند.
2. **پاک‌سازی داده‌ها**:
   - دکمه "پاک‌سازی داده‌ها" را فشار دهید تا مقادیر گمشده پر شوند و داده‌های پرت حذف شوند.
   - گزارش پاک‌سازی در یک پنجره نمایش داده می‌شود.
3. **داده‌کاوی**:
   - دکمه "داده‌کاوی" را فشار دهید تا آمار توصیفی، همبستگی‌ها و گزارش داده‌های پرت نمایش داده شود.
4. **مقایسه ستون‌ها**:
   - ۲ یا ۳ ستون را از لیست با Ctrl+کلیک انتخاب کنید.
   - دکمه "مقایسه ستون‌ها" را فشار دهید تا نمودار پراکندگی دوبعدی یا سه‌بعدی نمایش داده شود.
5. **نمایش پراکندگی داده‌ها**:
   - دکمه "نمودار پراکندگی" را فشار دهید تا نمودارهای جفتی برای حداکثر ۴ ستون عددی نمایش داده شود.
6. **پیش‌بینی**:
   - یک ستون هدف از منوی کشویی انتخاب کنید.
   - دکمه "شروع پیش‌بینی" را فشار دهید تا مدل پیشنهادی Gemini آموزش داده شود و گراف خطی آینده‌نگرانه نمایش داده شود.

### API (FastAPI)
1. **آپلود فایل CSV**:
   ```bash
   curl -X POST -F "file=@your_file.csv" http://localhost:8000/upload_csv
   ```
   پاسخ شامل لیست ستون‌ها است.
2. **پاک‌سازی داده‌ها**:
   ```bash
   curl -X POST http://localhost:8000/clean_data
   ```
   پاسخ شامل گزارش پاک‌سازی است.
3. **داده‌کاوی**:
   ```bash
   curl -X GET http://localhost:8000/mine_data
   ```
   پاسخ شامل آمار توصیفی، ماتریس همبستگی، و گزارش داده‌های پرت است.
4. **نمایش نمودارهای اولیه**:
   ```bash
   curl -X GET http://localhost:8000/visualize_data --output plots.png
   ```
   نمودارها به‌صورت فایل PNG ذخیره می‌شوند.
5. **مقایسه ستون‌ها**:
   ```bash
   curl -X POST -F "selected_columns=col1" -F "selected_columns=col2" http://localhost:8000/compare_columns --output compare.png
   ```
   نمودار پراکندگی دوبعدی یا سه‌بعدی تولید می‌شود.
6. **پیش‌بینی**:
   ```bash
   curl -X POST -F "target_column=your_target_column" http://localhost:8000/predict
   ```
   پاسخ شامل پیام موفقیت، توضیحات Gemini، و لینک به گراف خطی (مثل `/plot/prediction_{target_column}.png`) است.
7. **مشاهده گراف**:
   ```bash
   curl -X GET http://localhost:8000/plot/prediction_{target_column}.png --output prediction.png
   ```

---

## Usage Guide
### Graphical Application (GUI)
1. **Load CSV File**:
   - Click the "Load CSV File" button and select a CSV file.
   - Columns appear in the target dropdown and comparison list.
2. **Data Cleaning**:
   - Click the "Clean Data" button to fill missing values and remove outliers.
   - A cleaning report is displayed in a dialog.
3. **Data Mining**:
   - Click the "Data Mining" button to view stats, correlations, and outlier reports.
4. **Column Comparison**:
   - Select 2 or 3 columns using Ctrl+click.
   - Click the "Compare Columns" button to display a 2D or 3D scatter plot.
5. **Data Scatter Plot**:
   - Click the "Show Scatter Plot" button to display pairplots for up to 4 numeric columns.
6. **Prediction**:
   - Select a target column from the dropdown.
   - Click the "Start Prediction" button to train the Gemini-recommended model and display a future-oriented line graph.

### API (FastAPI)
1. **Upload CSV File**:
   ```bash
   curl -X POST -F "file=@your_file.csv" http://localhost:8000/upload_csv
   ```
   Returns a list of column names.
2. **Clean Data**:
   ```bash
   curl -X POST http://localhost:8000/clean_data
   ```
   Returns a cleaning report.
3. **Data Mining**:
   ```bash
   curl -X GET http://localhost:8000/mine_data
   ```
   Returns stats, correlation matrix, and outlier report.
4. **Initial Visualizations**:
   ```bash
   curl -X GET http://localhost:8000/visualize_data --output plots.png
   ```
   Saves plots as a PNG file.
5. **Column Comparison**:
   ```bash
   curl -X POST -F "selected_columns=col1" -F "selected_columns=col2" http://localhost:8000/compare_columns --output compare.png
   ```
   Generates a 2D or 3D scatter plot.
6. **Prediction**:
   ```bash
   curl -X POST -F "target_column=your_target_column" http://localhost:8000/predict
   ```
   Returns a success message, Gemini recommendation, and a link to the line graph (e.g., `/plot/prediction_{target_column}.png`).
7. **View Plot**:
   ```bash
   curl -X GET http://localhost:8000/plot/prediction_{target_column}.png --output prediction.png
   ```

## نکات
- **فرمت فایل CSV**: فایل CSV باید حداقل یک ستون عددی (صحیح یا اعشاری) برای هدف و یک ستون عددی برای ویژگی‌ها داشته باشد. ستون‌های غیرعددی برای مدل‌سازی نادیده گرفته می‌شوند.
- **پشتیبانی از فارسی**: نصب فونت Vazir و کتابخانه‌های `arabic-reshaper` و `python-bidi` برای نمایش صحیح متن فارسی ضروری است.
- **عملکرد**: برای مجموعه داده‌های بزرگ، تولید نمودارها یا پیش‌بینی ممکن است زمان‌بر باشد. در صورت نیاز، داده‌ها را نمونه‌برداری کنید.
- **مدیریت خطاها**: خطاها در `app_errors.log` ثبت می‌شوند. در GUI، پیام‌های خطا در نوار وضعیت و پنجره‌های پیام نمایش داده می‌شوند. در API، خطاها به‌صورت HTTPException برگردانده می‌شوند.
- **پیش‌بینی آینده**: گراف خطی در API شامل ۵ نقطه فرضی برای پیش‌بینی آینده است (خط سبز نقطه‌چین). برای تنظیم تعداد نقاط یا استفاده از ستون زمانی، کد را اصلاح کنید.

---

## Notes
- **CSV File Format**: The CSV file must include at least one numeric column (integer or float) for the target and one for features. Non-numeric columns are ignored for modeling.
- **Persian Text Support**: Install the Vazir font and `arabic-reshaper` and `python-bidi` libraries for proper Persian text rendering.
- **Performance**: For large datasets, plot generation or prediction may be slow. Sample the data if needed.
- **Error Handling**: Errors are logged in `app_errors.log`. In the GUI, errors appear in the status bar and message boxes. In the API, errors are returned as HTTPExceptions.
- **Future Prediction**: The API line graph includes 5 hypothetical future points (green dashed line). Modify the code to adjust the number of points or use a time-based column.

## توسعه‌دهندگان
برای افزودن قابلیت‌های جدید یا بهبود پروژه:
- **مقایسه چند مدل**: نمایش معیارهای تمام مدل‌ها در GUI برای مقایسه دقیق‌تر.
- **پشتیبانی از داده‌های دسته‌ای**: افزودن رمزگذاری (مانند one-hot encoding) برای ستون‌های غیرعددی.
- **بهبود رابط کاربری**: افزودن گزینه‌های سفارشی‌سازی برای رنگ و اندازه نمودارها در GUI.
- **ذخیره‌سازی خروجی**: امکان ذخیره نتایج پیش‌بینی و نمودارها به‌صورت فایل در GUI و ذخیره دائمی‌تر در API (مثل S3).
- **چرخش لاگ‌ها**: استفاده از `RotatingFileHandler` برای مدیریت اندازه فایل `app_errors.log`.
- **ستون زمانی در API**: استفاده از ستون زمانی (مثل تاریخ) برای محور X گراف خطی.

---

## Developers
To add new features or improve the project:
- **Multi-Model Comparison**: Display metrics for all models in the GUI for detailed comparison.
- **Categorical Data Support**: Implement encoding (e.g., one-hot encoding) for non-numeric columns.
- **UI Enhancements**: Add customization options for plot colors and sizes in the GUI.
- **Output Saving**: Add the ability to save prediction results and plots in the GUI and more permanent storage in the API (e.g., S3).
- **Log Rotation**: Use `RotatingFileHandler` to manage the size of `app_errors.log`.
- **Time-Based Axis in API**: Use a time-based column (e.g., date) for the X-axis of the line graph.

## مجوز
این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر، فایل `LICENSE` را بررسی کنید.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.