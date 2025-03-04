import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Manually create the DataFrame
data = pd.DataFrame({
    'StudyDuration': [100, 0, 14, 42, 90, 20, 38, 18, 22, 10, 10, 23, 35, 39, 23, 2, 21, 1, 23, 43, 29, 37, 1, 20, 32, 11, 21, 43, 24, 48, 26, 41, 27, 15, 14, 46, 50, 43, 2, 36, 50, 6, 20, 8, 38, 17, 3, 24, 13, 49],
    'ExamScore': [94, 83, 100, 10, 67, 89, 91, 75, 64, 57, 57, 92, 95, 82, 55, 41, 76, 54, 85, 83, 90, 94, 55, 80, 100, 73, 76, 100, 69, 97, 62, 85, 88, 58, 71, 100, 100, 100, 43, 92, 100, 45, 83, 76, 100, 53, 56, 77, 70, 100],
    'Attendance': [6, 14, 13, 6, 10, 8, 14, 14, 9, 9, 11, 12, 2, 14, 15, 6, 0, 3, 12, 15, 3, 13, 4, 6, 6, 12, 14, 10, 12, 3, 12, 15, 6, 10, 2, 5, 11, 1, 9, 12, 13, 8, 4, 5, 11, 11, 11, 11, 3, 13],
    'ExtraCurricular': ['YES', 'NO', 'YES', 'YES', 'NO', 'YES', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'YES', 'YES', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'YES', 'YES', 'YES', 'NO', 'NO', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'NO', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'NO', 'NO'],
    'Age': [22, 18, 25, 18, 20, 18, 19, 19, 21, 23, 24, 22, 25, 18, 18, 20, 23, 19, 22, 21, 19, 25, 23, 25, 24, 21, 24, 20, 25, 20, 18, 24, 23, 25, 22, 21, 19, 23, 23, 20, 24, 25, 25, 18, 18, 21, 20, 23, 25, 22]
})

# 1. Variable Identification
print("Variable Types:")
print("Study Duration: Quantitative (Discrete) - Independent")
print("Exam Score: Quantitative (Discrete) - Dependent")
print("Attendance: Quantitative (Discrete) - Independent")
print("Extra Curricular: Qualitative (Nominal) - Independent")
print("Age: Quantitative (Discrete) - Independent")

# 2a. Bar Chart for Study Duration and Exam Score
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
data['StudyDuration'].plot(kind='bar')
plt.title('Study Duration')
plt.xlabel('Student Index')
plt.ylabel('Hours')
plt.subplot(1,2,2)
data['ExamScore'].plot(kind='bar')
plt.title('Exam Score')
plt.xlabel('Student Index')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

# 2b. Frequency Table
def create_frequency_table(series):
    freq_table = series.value_counts().sort_index()
    return pd.DataFrame({
        'Value': freq_table.index,
        'Frequency': freq_table.values,
        'Relative Frequency': freq_table.values / len(series)
    })

study_duration_freq = create_frequency_table(data['StudyDuration'])
exam_score_freq = create_frequency_table(data['ExamScore'])

print("\nStudy Duration Frequency Table:")
print(study_duration_freq)
print("\nExam Score Frequency Table:")
print(exam_score_freq)

# 2c. Pie Charts
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
study_duration_freq['Relative Frequency'].plot(kind='pie', autopct='%1.1f%%')
plt.title('Study Duration Relative Frequency')
plt.subplot(1,2,2)
exam_score_freq['Relative Frequency'].plot(kind='pie', autopct='%1.1f%%')
plt.title('Exam Score Relative Frequency')
plt.tight_layout()
plt.savefig('pie_chart.png')
plt.close()

# 2d. Histogram
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
data['StudyDuration'].hist()
plt.title('Study Duration Histogram')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.subplot(1,2,2)
data['ExamScore'].hist()
plt.title('Exam Score Histogram')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

# 3. Central Tendency
print("\nCentral Tendency:")
print("Study Duration:")
print(f"Mean: {data['StudyDuration'].mean():.2f}")
print(f"Median: {data['StudyDuration'].median():.2f}")
print(f"Mode: {data['StudyDuration'].mode()[0]}")

print("\nExam Score:")
print(f"Mean: {data['ExamScore'].mean():.2f}")
print(f"Median: {data['ExamScore'].median():.2f}")
print(f"Mode: {data['ExamScore'].mode()[0]}")

# 4. Data Variability
print("\nData Variability:")
print("Study Duration:")
print(f"Range: {data['StudyDuration'].max() - data['StudyDuration'].min()}")
print(f"Variance: {data['StudyDuration'].var():.2f}")
print(f"Standard Deviation: {data['StudyDuration'].std():.2f}")

print("\nExam Score:")
print(f"Range: {data['ExamScore'].max() - data['ExamScore'].min()}")
print(f"Variance: {data['ExamScore'].var():.2f}")
print(f"Standard Deviation: {data['ExamScore'].std():.2f}")

# 5. Box Plot
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.boxplot(x=data['StudyDuration'])
plt.title('Study Duration Box Plot')
plt.subplot(1,2,2)
sns.boxplot(x=data['ExamScore'])
plt.title('Exam Score Box Plot')
plt.tight_layout()
plt.savefig('boxplot.png')
plt.close()

# Outlier Detection
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_below = series[series < lower_bound]
    outliers_above = series[series > upper_bound]
    return outliers_below, outliers_above

study_duration_outliers_below, study_duration_outliers_above = detect_outliers(data['StudyDuration'])
exam_score_outliers_below, exam_score_outliers_above = detect_outliers(data['ExamScore'])

print("\nOutliers:")
print("Study Duration Outliers Below:", list(study_duration_outliers_below))
print("Study Duration Outliers Above:", list(study_duration_outliers_above))
print("Exam Score Outliers Below:", list(exam_score_outliers_below))
print("Exam Score Outliers Above:", list(exam_score_outliers_above))