
# 🎓 Student Exam Score Prediction Project

This project aims to predict students' **exam scores** based on their lifestyle and study habits using various regression models and GridSearchCV for hyperparameter tuning.

---

## 📂 Dataset

- The dataset is read from a CSV file: `student_habits_performance.csv`
- It includes features like:
  - Study hours
  - Attendance
  - Sleep hours
  - Mental health rating
  - Parental education
  - Screen time (Netflix + Social Media)
  - Diet quality
  - Internet quality
  - Exercise frequency
  - Part-time job status

---

## 🔧 Libraries Used

```python
pandas, numpy, matplotlib, seaborn  
sklearn (preprocessing, models, metrics, GridSearchCV)  
xgboost  
pickle
```

---

## 📊 Steps Followed

### 1. Data Cleaning
- Dropped `student_id` (non-informative).
- Removed duplicates.
- Filled missing values in `parental_education_level` using the most frequent value.
- Handled irregular values (e.g., `"Other"` in `gender` replaced with `"Male"`).

### 2. Data Visualization
- Count plots and box plots for all categorical variables.
- Histograms, boxen plots, and scatter plots for numerical features.
- Pairplot to see relationships.
- Heatmap for correlation matrix.

### 3. Outlier Handling
- Used IQR method for all numerical columns to remove outliers.

### 4. Feature Engineering
- Created a new feature `wasted_time = social_media_hours + netflix_hours`.
- Removed original `social_media_hours` and `netflix_hours`.
- Dropped `age` and low-impact features (`gender`, `extracurricular_participation`) after feature selection.

---

## 🧠 Modeling

### 1. Data Preparation
- Standardized numerical features using `StandardScaler`.
- Encoded categorical features using `LabelEncoder`.
- Split data into `X` and `y`, and then into train/test (80/20 split).

### 2. Feature Selection
Used `SelectKBest(f_regression)` to analyze which categorical columns are most predictive.

### 3. Regression Models
Trained with **GridSearchCV** for hyperparameter tuning:
- Linear Regression
- SVR
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- KNN
- XGBoost

### 4. Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Results were visualized using bar charts for comparison.

---

## 🏆 Best Performing Models

A summary DataFrame was created, ranking models by their R² scores.

---

## 💾 Saving Models

Saved the following with `pickle`:
- `best_model.pkl`: Final model (Linear Regression)
- `label_encoders.pkl`: Encoded object columns
- `scaler.pkl`: Scaler used for numeric features

---

## 📬 Author

Mazen Emad — Student Exam Score Prediction Project  
Contact for more insights or collaborations.

