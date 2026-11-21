# Student Performance and Attendance Analysis
# Made by: <your name>, B.Tech 1st year, ABES Engineering College
# HCL Training Project

import numpy as np
import pandas as pd

# ---------- 1. Loading the data ----------
df = pd.read_csv("/content/student_performance_data.csv")
print("Rows and columns:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# ---------- 2. Cleaning the data ----------
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# text columns -> most common value, number columns -> median
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

before = len(df)
df = df.drop_duplicates()
print("\nDuplicates removed:", before - len(df))
print("Missing values left:", df.isnull().sum().sum())

# ---------- 3. Overall numbers using numpy ----------
scores = df["Exam_Score"].to_numpy()

print("\n--- Overall Exam Score ---")
print("Average:", round(np.mean(scores), 2))
print("Median:", np.median(scores))
print("Highest:", np.max(scores))
print("Lowest:", np.min(scores))
print("Standard deviation:", round(np.std(scores), 2))
print("Average attendance:", round(df["Attendance"].mean(), 2), "%")

# ---------- 4. Attendance groups vs marks ----------
df["Attendance_Group"] = pd.cut(
    df["Attendance"],
    bins=[0, 70, 80, 90, 100],
    labels=["Below 70", "70-80", "80-90", "90-100"]
)

attendance_result = (
    df.groupby("Attendance_Group", observed=True)["Exam_Score"]
    .agg(["count", "mean", "max", "min"])
    .round(2)
)
print("\n--- Attendance Group vs Exam Score ---")
print(attendance_result)

# ---------- 5. Correlation ----------
cols = ["Attendance", "Hours_Studied", "Sleep_Hours",
        "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]
corr = df[cols + ["Exam_Score"]].corr()["Exam_Score"].drop("Exam_Score")
corr = corr.sort_values(ascending=False)

print("\n--- Correlation with Exam Score ---")
print(corr.round(3))
print("Numpy check (attendance vs score):",
      round(np.corrcoef(df["Attendance"], df["Exam_Score"])[0, 1], 3))

# ---------- 6. Pass/Fail and grades ----------
# I am taking 60 as the pass mark
df["Result"] = np.where(df["Exam_Score"] >= 60, "Pass", "Fail")

conditions = [
    df["Exam_Score"] >= 80,
    df["Exam_Score"] >= 70,
    df["Exam_Score"] >= 60,
]
df["Grade"] = np.select(conditions, ["A", "B", "C"], default="D")

print("\n--- Pass / Fail count ---")
print(df["Result"].value_counts())
print("\n--- Grade count ---")
print(df["Grade"].value_counts().sort_index())

pass_rate = df.groupby("Attendance_Group", observed=True)["Result"].apply(
    lambda x: round((x == "Pass").mean() * 100, 2)
)
print("\n--- Pass percentage in each attendance group ---")
print(pass_rate)

# ---------- 7. Other factors ----------
df["Study_Group"] = pd.cut(
    df["Hours_Studied"],
    bins=[0, 10, 20, 30, 50],
    labels=["0-10", "10-20", "20-30", "30+"]
)
print("\n--- Study hours vs average score ---")
print(df.groupby("Study_Group", observed=True)["Exam_Score"].mean().round(2))

print("\n--- Gender wise average score ---")
print(df.groupby("Gender")["Exam_Score"].mean().round(2))

show = ["Attendance", "Hours_Studied", "Exam_Score"]
print("\nTop 5 students:")
print(df.sort_values("Exam_Score", ascending=False).head(5)[show])
print("\nBottom 5 students:")
print(df.sort_values("Exam_Score").head(5)[show])

# ---------- 8. Conclusion (written from the actual results) ----------
best = attendance_result["mean"].idxmax()
low_avg = attendance_result["mean"].iloc[0]
high_avg = attendance_result["mean"].iloc[-1]

print("\n========== CONCLUSION ==========")
print(f"1. Attendance and exam score correlation is {corr['Attendance']:.2f}.")
print(f"2. Students with 90-100% attendance scored {high_avg:.2f} on average, "
      f"while students below 70% scored {low_avg:.2f}.")
print(f"3. Hours studied correlation is {corr['Hours_Studied']:.2f}, "
      "so studying more also helps.")
print(f"4. Pass percentage in the highest attendance group is {pass_rate.iloc[-1]}%.")
print(f"5. The group with the best average marks is: {best}.")

# saving the cleaned file
df.to_csv("cleaned_student_data.csv", index=False)
print("\nCleaned file saved as cleaned_student_data.csv")