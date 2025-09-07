# پیش‌بینی مالی پیشرفته (Advanced Financial Predictor)

## معرفی
این پروژه یک برنامه گرافیکی (GUI) مبتنی بر PyQt5 است که برای تحلیل داده‌های مالی و پیش‌بینی مقادیر عددی با استفاده از مدل‌های یادگیری ماشین طراحی شده است. این برنامه امکان بارگذاری فایل‌های CSV، پاک‌سازی داده‌ها، داده‌کاوی، مقایسه ستون‌ها و پیش‌بینی خودکار با بهترین مدل یادگیری ماشین را فراهم می‌کند. همچنین، قابلیت نمایش نمودارهای مختلف مانند پراکندگی، هیستوگرام، جعبه‌ای و نقشه حرارتی همبستگی را دارد. خطاها در فایل `app_errors.log` ثبت می‌شوند تا از کرش برنامه جلوگیری شود.

---

## Introduction
This project is a PyQt5-based graphical user interface (GUI) application designed for financial data analysis and prediction using machine learning models. It allows users to load CSV files, clean data, perform data mining, compare columns, and automatically predict using the best-performing machine learning model. It also provides visualizations such as scatter plots, histograms, box plots, and correlation heatmaps. Errors are logged in `app_errors.log` to prevent application crashes.

## ویژگی‌ها
- **بارگذاری داده‌ها**: بارگذاری فایل‌های CSV برای تحلیل.
- **پاک‌سازی داده‌ها**: پر کردن مقادیر گمشده و حذف داده‌های پرت با روش IQR.
- **داده‌کاوی**: ارائه آمار توصیفی، ماتریس همبستگی و گزارش داده‌های پرت.
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
- **مدیریت خطاها**: ثبت خطاها در فایل `app_errors.log` برای جلوگیری از کرش.
- **رابط کاربری بهبودیافته**: استفاده از QGridLayout، نوار وضعیت، انتخاب چندگانه ستون‌ها و استایل‌دهی مدرن.

---

## Features
- **Data Loading**: Load CSV files for analysis.
- **Data Cleaning**: Fill missing values and remove outliers using the IQR method.
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
- **Error Handling**: Log errors in `app_errors.log` to prevent crashes.
- **Enhanced UI**: Uses QGridLayout, status bar, multi-column selection, and modern styling.

## پیش‌نیازها
برای اجرای این برنامه، نیاز به نصب کتابخانه‌های زیر دارید:

```bash
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost arabic-reshaper python-bidi
```

**توجه**: کتابخانه `xgboost` اختیاری است. اگر نصب نشود، گزینه XGBoost غیرفعال خواهد بود.

### پشتیبانی از متن فارسی
برای نمایش صحیح متن فارسی در رابط کاربری و نمودارها، نصب فونت Vazir و کتابخانه‌های زیر ضروری است:

```bash
pip install arabic-reshaper python-bidi
```

1. فونت Vazir را از [اینجا](https://github.com/rastikerdar/vazir-font) دانلود کنید.
2. فونت را در سیستم نصب کنید:
   - **ویندوز**: فایل `.ttf` را به `C:\Windows\Fonts` کپی کنید.
   - **لینوکس/مک**: مراحل نصب فونت سیستم را دنبال کنید.
3. تنظیمات Matplotlib را برای استفاده از فونت Vazir به‌روزرسانی کنید:

```python
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = "path/to/Vazir.ttf"  # مسیر فایل فونت
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Vazir"
```

---

## Requirements
To run this application, install the following libraries:

```bash
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost arabic-reshaper python-bidi
```

**Note**: The `xgboost` library is optional. If not installed, the XGBoost option will be unavailable.

### Persian Text Support
To properly render Persian text in the GUI and plots, install the Vazir font and the following libraries:

```bash
pip install arabic-reshaper python-bidi
```

1. Download the Vazir font from [here](https://github.com/rastikerdar/vazir-font).
2. Install the font on your system:
   - **Windows**: Copy the `.ttf` file to `C:\Windows\Fonts`.
   - **Linux/macOS**: Follow system-specific font installation steps.
3. Update Matplotlib settings to use the Vazir font:

```python
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = "path/to/Vazir.ttf"  # Path to the font file
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Vazir"
```

## ساختار پروژه
پروژه به‌صورت ماژولار سازمان‌دهی شده و شامل فایل‌های زیر است:

- `main.py`: نقطه ورود برنامه، راه‌اندازی اپلیکیشن PyQt5.
- `ui.py`: تنظیم رابط کاربری و ویجت‌ها.
- `data_processing.py`: منطق بارگذاری، پاک‌سازی و داده‌کاوی.
- `visualization.py`: تولید نمودارهای اولیه، پراکندگی و مقایسه.
- `prediction.py`: آموزش خودکار بهترین مدل، پیش‌بینی و نمایش نتایج.
- `dialog.py`: دیالوگ برای نمایش گزارش‌ها.
- `app_errors.log`: فایل لاگ برای ثبت خطاها.

---

## Project Structure
The project is organized modularly and includes the following files:

- `main.py`: Entry point, initializes the PyQt5 application.
- `ui.py`: Sets up the user interface and widgets.
- `data_processing.py`: Handles data loading, cleaning, and mining.
- `visualization.py`: Generates initial, scatter, and comparison plots.
- `prediction.py`: Manages automatic model selection, prediction, and result display.
- `dialog.py`: Dialog for displaying reports.
- `app_errors.log`: Log file for recording errors.

## توابع پروژه
در این بخش، تمام توابع موجود در فایل‌های پروژه و عملکرد آن‌ها شرح داده شده است. همچنین، مشخص شده که هر تابع به کدام بخش از رابط کاربری (در صورت وجود) متصل است.

### **فایل `main.py`**
- **تابع `__main__`**:
  - **عملکرد**: نقطه ورود برنامه. یک نمونه از `QApplication` و کلاس `AdvancedFinancialPredictorUI` ایجاد می‌کند، پنجره اصلی را نمایش می‌دهد و برنامه را اجرا می‌کند. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به هیچ بخش خاصی از رابط کاربری متصل نیست؛ مستقیماً با اجرای `python main.py` فراخوانی می‌شود.

### **فایل `ui.py`**
- **تابع `__init__`** (در کلاس `AdvancedFinancialPredictorUI`):
  - **عملکرد**: رابط کاربری گرافیکی را تنظیم می‌کند، شامل دکمه‌ها، منوی کشویی، لیست مقایسه (با قابلیت انتخاب چندگانه)، جدول نتایج، ناحیه نمودار و نوار وضعیت. سیگنال‌های دکمه‌ها را به توابع مربوطه متصل می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `AdvancedFinancialPredictorUI` در `main.py` فراخوانی می‌شود.

### **فایل `data_processing.py`**
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

### **فایل `visualization.py`**
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

### **فایل `prediction.py`**
کلاس `Predictor` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: نمونه‌ای از کلاس را ایجاد کرده و شیء والد (`parent`) را ذخیره می‌کند.
  - **اتصال**: هنگام ایجاد نمونه `Predictor` در `ui.py` فراخوانی می‌شود.
- **تابع `train_and_predict`**:
  - **عملکرد**: داده‌ها را آماده می‌کند (استانداردسازی، تقسیم به مجموعه آموزش و آزمون)، تمام مدل‌های موجود را آزمایش می‌کند، بهترین مدل را بر اساس R² انتخاب می‌کند، پیش‌بینی انجام می‌دهد و نتایج (MAE، RMSE، R²) را در جدول و نمودار نمایش می‌دهد. خطاها در `app_errors.log` ثبت می‌شوند.
  - **اتصال**: به دکمه "شروع پیش‌بینی" (`btn_predict`) متصل است.

### **فایل `dialog.py`**
کلاس `DataAnalysisDialog` شامل توابع زیر است:
- **تابع `__init__`**:
  - **عملکرد**: یک دیالوگ برای نمایش گزارش‌ها (مثل گزارش پاک‌سازی یا داده‌کاوی) با کادر متنی و دکمه بستن ایجاد می‌کند.
  - **اتصال**: توسط `clean_data` و `mine_data` در `data_processing.py` فراخوانی می‌شود.

---

## Project Functions
This section lists all functions in the project files and describes their purpose and connections to the user interface (if applicable).

### **File `main.py`**
- **Function `__main__`**:
  - **Purpose**: Entry point of the program. Creates a `QApplication` instance and an `AdvancedFinancialPredictorUI` instance, displays the main window, and runs the application. Errors are logged in `app_errors.log`.
  - **Connection**: Not connected to any UI element; invoked by running `python main.py`.

### **File `ui.py`**
- **Function `__init__`** (in class `AdvancedFinancialPredictorUI`):
  - **Purpose**: Sets up the graphical user interface, including buttons, dropdown, comparison list (with multi-selection), results table, plot area, and status bar. Connects button signals to corresponding functions.
  - **Connection**: Called when creating an instance of `AdvancedFinancialPredictorUI` in `main.py`.

### **File `data_processing.py`**
The `DataProcessor` class includes the following functions:
- **Function `__init__`**:
  - **Purpose**: Initializes the class and stores the parent object.
  - **Connection**: Called when creating a `DataProcessor` instance in `ui.py`.
- **Function `load_csv`**:
  - **Purpose**: Loads a CSV file, populates the target dropdown and comparison list with column names, and triggers initial visualizations. Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Load CSV File" button (`btn_load`).
- **Function `clean_data`**:
  - **Purpose**: Cleans data by filling missing values (mean for numeric, mode for non-numeric) and removing outliers using the IQR method. Displays a cleaning report. Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Clean Data" button (`btn_clean`).
- **Function `mine_data`**:
  - **Purpose**: Performs data mining, generating descriptive statistics, a correlation matrix, and an outlier report, displayed in a dialog. Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Data Mining" button (`btn_mine`).

### **File `visualization.py`**
The `DataVisualizer` class includes the following functions:
- **Function `__init__`**:
  - **Purpose**: Initializes the class and stores the parent object.
  - **Connection**: Called when creating a `DataVisualizer` instance in `ui.py`.
- **Function `visualize_data`**:
  - **Purpose**: Generates initial plots (scatter, histogram, box plot, and correlation heatmap) for numeric columns. Errors are logged in `app_errors.log`.
  - **Connection**: Called after loading a CSV file (`load_csv`) or cleaning data (`clean_data`).
- **Function `show_scatter_plot`**:
  - **Purpose**: Generates pairplots for up to 4 numeric columns. Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Show Scatter Plot" button (`btn_scatter`).
- **Function `compare_columns`**:
  - **Purpose**: Generates a 2D scatter plot (for 2 columns) or a 3D scatter plot (for 3 columns) for selected columns. Supports multi-selection (Ctrl+click). Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Compare Columns" button (`btn_compare`).

### **File `prediction.py`**
The `Predictor` class includes the following functions:
- **Function `__init__`**:
  - **Purpose**: Initializes the class and stores the parent object.
  - **Connection**: Called when creating a `Predictor` instance in `ui.py`.
- **Function `train_and_predict`**:
  - **Purpose**: Prepares data (standardization, train-test split), tests all available models, selects the best model based on R², performs predictions, and displays results (MAE, RMSE, R²) in a table and scatter plot. Errors are logged in `app_errors.log`.
  - **Connection**: Connected to the "Start Prediction" button (`btn_predict`).

### **File `dialog.py`**
The `DataAnalysisDialog` class includes the following functions:
- **Function `__init__`**:
  - **Purpose**: Creates a dialog to display reports (e.g., cleaning or mining reports) with a text box and a close button.
  - **Connection**: Called by `clean_data` and `mine_data` in `data_processing.py`.

## نصب و اجرا
1. مخزن پروژه را کلون کنید یا فایل‌ها را در یک دایرکتوری ذخیره کنید.
2. کتابخانه‌های مورد نیاز را نصب کنید (بخش پیش‌نیازها).
3. برنامه را با اجرای فایل `main.py` اجرا کنید:

```bash
python main.py
```

---

## Installation and Running
1. Clone the repository or save the files in a directory.
2. Install the required libraries (see Requirements).
3. Run the application by executing `main.py`:

```bash
python main.py
```

## راهنمای استفاده
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
   - دکمه "شروع پیش‌بینی" را فشار دهید تا بهترین مدل به طور خودکار انتخاب شود و نتایج (MAE، RMSE، R²) در جدول و نمودار نمایش داده شود.

---

## Usage Guide
1. **Load CSV File**:
   - Click the "Load CSV File" button and select a CSV file.
   - The file's columns will appear in the target dropdown and comparison list.
2. **Data Cleaning**:
   - Click the "Clean Data" button to fill missing values and remove outliers.
   - A cleaning report will be displayed in a dialog.
3. **Data Mining**:
   - Click the "Data Mining" button to view descriptive statistics, correlation matrix, and outlier report.
4. **Column Comparison**:
   - Select 2 or 3 columns from the list using Ctrl+click.
   - Click the "Compare Columns" button to display a 2D or 3D scatter plot.
5. **Data Scatter Plot**:
   - Click the "Show Scatter Plot" button to display pairplots for up to 4 numeric columns.
6. **Prediction**:
   - Select a target column from the dropdown.
   - Click the "Start Prediction" button to automatically select the best model and display results (MAE, RMSE, R²) in a table and scatter plot.

## نکات
- **فرمت فایل CSV**: فایل CSV باید شامل ستون‌های عددی برای پیش‌بینی و مقایسه باشد. ستون‌های غیرعددی برای مدل‌سازی نادیده گرفته می‌شوند.
- **پشتیبانی از فارسی**: نصب فونت Vazir و کتابخانه‌های `arabic-reshaper` و `python-bidi` برای نمایش صحیح متن فارسی ضروری است.
- **عملکرد**: برای مجموعه داده‌های بزرگ، تولید نمودارها ممکن است زمان‌بر باشد. در صورت نیاز، داده‌ها را نمونه‌برداری کنید.
- **مدیریت خطاها**: خطاها در فایل `app_errors.log` ثبت می‌شوند و پیام‌های خطا در نوار وضعیت و پنجره‌های پیام نمایش داده می‌شوند.

---

## Notes
- **CSV File Format**: The CSV file must include numeric columns for prediction and comparison. Non-numeric columns are ignored for modeling.
- **Persian Text Support**: Install the Vazir font and `arabic-reshaper` and `python-bidi` libraries for proper Persian text rendering.
- **Performance**: For large datasets, plot generation may be slow. Consider sampling the data if needed.
- **Error Handling**: Errors are logged in `app_errors.log`, and error messages are displayed in the status bar and message boxes.

## توسعه‌دهندگان
برای افزودن قابلیت‌های جدید یا بهبود برنامه:
- **مقایسه چند مدل**: نمایش معیارهای تمام مدل‌ها در یک دیالوگ برای مقایسه دقیق‌تر.
- **پشتیبانی از داده‌های دسته‌ای**: افزودن رمزگذاری (مانند one-hot encoding) برای پشتیبانی از ستون‌های غیرعددی.
- **بهبود رابط کاربری**: افزودن گزینه‌های سفارشی‌سازی برای رنگ و اندازه نمودارها، یا نمایش تعداد ستون‌های انتخاب‌شده.
- **ذخیره‌سازی خروجی**: امکان ذخیره نتایج پیش‌بینی و نمودارها به صورت فایل.
- **چرخش لاگ‌ها**: استفاده از `RotatingFileHandler` برای مدیریت اندازه فایل `app_errors.log`.

---

## Developers
To add new features or improve the application:
- **Multi-Model Comparison**: Display metrics for all models in a dialog for detailed comparison.
- **Categorical Data Support**: Implement encoding (e.g., one-hot encoding) to support non-numeric columns.
- **UI Enhancements**: Add customization options for plot colors and sizes, or display the number of selected columns.
- **Output Saving**: Add the ability to save prediction results and plots to files.
- **Log Rotation**: Use `RotatingFileHandler` to manage the size of `app_errors.log`.

## مجوز
این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر، فایل `LICENSE` را بررسی کنید.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.