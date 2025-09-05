# پیش‌بینی مالی پیشرفته (Advanced Financial Predictor)

## معرفی
این پروژه یک برنامه گرافیکی (GUI) مبتنی بر PyQt5 است که برای تحلیل داده‌های مالی و پیش‌بینی مقادیر عددی با استفاده از مدل‌های یادگیری ماشین طراحی شده است. این برنامه امکان بارگذاری فایل‌های CSV، پاک‌سازی داده‌ها، داده‌کاوی، مقایسه ستون‌ها و پیش‌بینی با مدل‌های مختلف را فراهم می‌کند. همچنین، قابلیت نمایش نمودارهای مختلف مانند پراکندگی، هیستوگرام، جعبه‌ای و نقشه حرارتی همبستگی را دارد.

---

## Introduction
This project is a PyQt5-based graphical user interface (GUI) application designed for financial data analysis and prediction using machine learning models. It allows users to load CSV files, clean data, perform data mining, compare columns, and make predictions with various models. It also provides visualizations such as scatter plots, histograms, box plots, and correlation heatmaps.

## ویژگی‌ها
- **بارگذاری داده‌ها**: بارگذاری فایل‌های CSV برای تحلیل.
- **پاک‌سازی داده‌ها**: پر کردن مقادیر گمشده و حذف داده‌های پرت با روش IQR.
- **داده‌کاوی**: ارائه آمار توصیفی، ماتریس همبستگی و گزارش داده‌های پرت.
- **مقایسه ستون‌ها**: نمایش نمودار پراکندگی دوبعدی یا سه‌بعدی برای ۲ یا ۳ ستون انتخابی.
- **نمایش پراکندگی داده‌ها**: نمایش نمودارهای جفتی (Pairplot) برای ستون‌های عددی.
- **پیش‌بینی**: آموزش و ارزیابی مدل‌های یادگیری ماشین شامل:
  - رگرسیون خطی (Linear Regression)
  - جنگل تصادفی (Random Forest Regressor)
  - درخت تصمیم (Decision Tree Regressor)
  - تقویت گرادیان (Gradient Boosting Regressor)
  - ماشین بردار پشتیبان (Support Vector Regressor)
  - XGBoost (در صورت نصب)
- **نمایش نتایج**: نمایش مقادیر واقعی و پیش‌بینی‌شده در جدول و نمودار پراکندگی.
- **رابط کاربری بهبودیافته**: استفاده از QGridLayout، نوار وضعیت و استایل‌دهی مدرن.

---

## Features
- **Data Loading**: Load CSV files for analysis.
- **Data Cleaning**: Fill missing values and remove outliers using the IQR method.
- **Data Mining**: Provide descriptive statistics, correlation matrix, and outlier reports.
- **Column Comparison**: Display 2D or 3D scatter plots for 2 or 3 selected columns.
- **Data Scatter Plot**: Show pairplots for numeric columns.
- **Prediction**: Train and evaluate machine learning models, including:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor
  - XGBoost (if installed)
- **Result Visualization**: Display actual vs. predicted values in a table and scatter plot.
- **Enhanced UI**: Uses QGridLayout, status bar, and modern styling.

## پیش‌نیازها
برای اجرای این برنامه، نیاز به نصب کتابخانه‌های زیر دارید:

```bash
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost
```

**توجه**: کتابخانه `xgboost` اختیاری است. اگر نصب نشود، گزینه XGBoost در برنامه غیرفعال خواهد بود.

### پشتیبانی از متن فارسی
برای نمایش صحیح متن فارسی در رابط کاربری و نمودارها، نصب کتابخانه‌های زیر و یک فونت فارسی (مانند Vazir) توصیه می‌شود:

```bash
pip install arabic-reshaper python-bidi
```

1. فونت Vazir را از [اینجا](https://github.com/rastikerdar/vazir-font) دانلود کنید.
2. فونت را در سیستم نصب کنید:
   - **ویندوز**: فایل `.ttf` را به `C:\Windows\Fonts` کپی کنید.
   - **لینوکس/مک**: مراحل نصب فونت سیستم را دنبال کنید.
3. برای استفاده از فونت Vazir در نمودارها، تنظیمات Matplotlib را به‌روزرسانی کنید:

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
pip install pandas numpy scikit-learn PyQt5 matplotlib seaborn xgboost
```

**Note**: The `xgboost` library is optional. If not installed, the XGBoost option will be unavailable.

### Persian Text Support
To properly render Persian text in the GUI and plots, install the following libraries and a Persian font (e.g., Vazir):

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
پروژه به‌صورت ماژولار سازمان‌دهی شده است و شامل فایل‌های زیر است:

- `main.py`: نقطه ورود برنامه، راه‌اندازی اپلیکیشن PyQt5.
- `ui.py`: تنظیمات رابط کاربری و ویجت‌ها.
- `data_processing.py`: منطق بارگذاری، پاک‌سازی و داده‌کاوی.
- `visualization.py`: تولید نمودارهای اولیه، پراکندگی و مقایسه.
- `prediction.py`: آموزش مدل، پیش‌بینی و نمایش نتایج.
- `dialog.py`: دیالوگ برای نمایش گزارش‌ها.

---

## Project Structure
The project is organized modularly and includes the following files:

- `main.py`: Entry point, initializes the PyQt5 application.
- `ui.py`: Sets up the user interface and widgets.
- `data_processing.py`: Handles data loading, cleaning, and mining.
- `visualization.py`: Generates initial, scatter, and comparison plots.
- `prediction.py`: Manages model training, prediction, and result display.
- `dialog.py`: Dialog for displaying reports.

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
   - ۲ یا ۳ ستون را از لیست انتخاب کنید (با Ctrl+کلیک).
   - دکمه "مقایسه ستون‌ها" را فشار دهید تا نمودار پراکندگی دوبعدی یا سه‌بعدی نمایش داده شود.
5. **نمایش پراکندگی داده‌ها**:
   - دکمه "نمودار پراکندگی" را فشار دهید تا نمودارهای جفتی برای حداکثر ۴ ستون عددی نمایش داده شود.
6. **پیش‌بینی**:
   - یک ستون هدف از منوی کشویی انتخاب کنید.
   - یک مدل پیش‌بینی (مانند رگرسیون خطی) را انتخاب کنید.
   - دکمه "شروع پیش‌بینی" را فشار دهید تا مدل آموزش ببیند و نتایج (MAE، RMSE، R²) در یک جدول و نمودار نمایش داده شود.

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
   - Select 2 or 3 columns from the list (Ctrl+click for multiple selections).
   - Click the "Compare Columns" button to display a 2D or 3D scatter plot.
5. **Data Scatter Plot**:
   - Click the "Show Scatter Plot" button to display pairplots for up to 4 numeric columns.
6. **Prediction**:
   - Select a target column from the dropdown.
   - Choose a prediction model (e.g., Linear Regression).
   - Click the "Start Prediction" button to train the model and view results (MAE, RMSE, R²) in a table and scatter plot.

## نکات
- **فرمت فایل CSV**: فایل CSV باید شامل ستون‌های عددی برای پیش‌بینی و مقایسه باشد. ستون‌های غیرعددی برای مدل‌سازی نادیده گرفته می‌شوند.
- **پشتیبانی از فارسی**: برای نمایش صحیح متن فارسی، نصب فونت Vazir و کتابخانه‌های `arabic-reshaper` و `python-bidi` ضروری است.
- **عملکرد**: برای مجموعه داده‌های بزرگ، ممکن است تولید نمودارها زمان‌بر باشد. در صورت نیاز، داده‌ها را نمونه‌برداری کنید.
- **خطاها**: در صورت بروز خطا، پیام‌های خطا در نوار وضعیت و پنجره‌های پیام نمایش داده می‌شوند.

---

## Notes
- **CSV File Format**: The CSV file must include numeric columns for prediction and comparison. Non-numeric columns are ignored for modeling.
- **Persian Text Support**: To properly display Persian text, install the Vazir font and the `arabic-reshaper` and `python-bidi` libraries.
- **Performance**: For large datasets, plot generation may be slow. Consider sampling the data if needed.
- **Error Handling**: Error messages are displayed in the status bar and message boxes if issues occur.

## توسعه‌دهندگان
برای افزودن قابلیت‌های جدید یا بهبود برنامه:
- **مقایسه چند مدل**: می‌توانید قابلیتی اضافه کنید که چندین مدل را به طور همزمان مقایسه کند.
- **پشتیبانی از داده‌های دسته‌ای**: اضافه کردن رمزگذاری (مانند one-hot encoding) برای پشتیبانی از ستون‌های غیرعددی.
- **بهبود رابط کاربری**: افزودن گزینه‌های سفارشی‌سازی برای رنگ و اندازه نمودارها.
- **ذخیره‌سازی خروجی**: افزودن امکان ذخیره نتایج پیش‌بینی و نمودارها به صورت فایل.

---

## Developers
To add new features or improve the application:
- **Multi-Model Comparison**: Add functionality to compare multiple models simultaneously.
- **Categorical Data Support**: Implement encoding (e.g., one-hot encoding) to support non-numeric columns.
- **UI Enhancements**: Add customization options for plot colors and sizes.
- **Output Saving**: Add the ability to save prediction results and plots to files.

## مجوز
این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر، فایل `LICENSE` را بررسی کنید.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.