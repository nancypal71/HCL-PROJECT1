"""
Hospital Appointment Operations Analysis (simple version)
Libraries used: pandas and numpy only.
Dataset: 09_hospital_appointments.csv (operational data, NOT clinical data)

Run: python hospital_analysis_simple.py
If the CSV is somewhere else, add its path to POSSIBLE_FILES below.
"""

import pandas as pd
import numpy as np

# The program tries these places one by one and uses the first file it finds.
# Add your own path to the list if needed.
POSSIBLE_FILES = [
    "09_hospital_appointments.csv",
    "data/09_hospital_appointments.csv",
    r"C:\Users\LAKSHYA\Documents\09_hospital_appointments.csv",
    r"C:\Users\LAKSHYA\Downloads\09_hospital_appointments.csv",
    r"C:\Users\LAKSHYA\Downloads\hospital_project_xyz\data\09_hospital_appointments.csv",
    r"C:\Users\LAKSHYA\Desktop\09_hospital_appointments.csv",
]

# ---------------------------------------------------------------
# TASK 124: Load and inspect appointment records
# ---------------------------------------------------------------
print("TASK 124: LOAD AND INSPECT")
df = None
for path in POSSIBLE_FILES:
    try:
        df = pd.read_csv(path)
        print("Loaded file:", path)
        break
    except FileNotFoundError:
        continue

# If the file was not found, ask the user to paste the full path
while df is None:
    print("CSV file not found in the usual places.")
    typed = input("Paste the full path of 09_hospital_appointments.csv: ")
    typed = typed.strip().strip('"').strip("'")
    if not typed.lower().endswith("09_hospital_appointments.csv"):
        print("Wrong file. The path must end with 09_hospital_appointments.csv")
        print("(not .zip and not .py). Extract the zip first, then open the data folder.")
        continue
    try:
        df = pd.read_csv(typed)
        print("Loaded file:", typed)
    except FileNotFoundError:
        print("Still not found, check the path and try again.")

print("Rows, columns:", df.shape)
print(df.head())
print(df.dtypes)
print(df.describe())

# ---------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------
# Step 1: remove extra spaces from text columns
for col in ["Appointment_ID", "Department", "Appointment_Type",
            "SMS_Reminder", "Attendance_Status"]:
    df[col] = df[col].str.strip()

# Step 2: convert date column to real dates
df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"])

# ---------------------------------------------------------------
# TASK 125: Missing waiting times
# ---------------------------------------------------------------
print("\nTASK 125: MISSING WAITING TIMES")
print("Missing values per column:")
print(df.isnull().sum())

# keep a flag so we know which rows were filled
df["Wait_Was_Missing"] = df["Wait_Time_Min"].isnull()

# fill with the median wait of the same department
df["Wait_Time_Min"] = df["Wait_Time_Min"].fillna(
    df.groupby("Department")["Wait_Time_Min"].transform("median"))
print("Missing after cleaning:", df["Wait_Time_Min"].isnull().sum())

# ---------------------------------------------------------------
# TASK 126: Duplicate Appointment_ID
# ---------------------------------------------------------------
print("\nTASK 126: DUPLICATE APPOINTMENT_ID")
dups = df["Appointment_ID"].duplicated().sum()
print("Duplicate IDs:", dups)
df = df.drop_duplicates(subset="Appointment_ID")   # does nothing if 0

# save cleaned file
df.to_csv("cleaned_hospital_appointments.csv", index=False)

# ---------------------------------------------------------------
# TASK 127: Total appointments and attendance-status percentages
# ---------------------------------------------------------------
print("\nTASK 127: TOTAL APPOINTMENTS AND STATUS %")
total = len(df)
print("Total appointments:", total)
status_pct = (df["Attendance_Status"].value_counts() / total * 100).round(2)
print(status_pct)

# ---------------------------------------------------------------
# TASK 128: Average waiting time by department
# ---------------------------------------------------------------
print("\nTASK 128: AVERAGE WAIT BY DEPARTMENT")
avg_wait = df.groupby("Department").agg(
    Appointments=("Appointment_ID", "count"),
    Avg_Wait_Min=("Wait_Time_Min", "mean")).round(2)
print(avg_wait.sort_values("Avg_Wait_Min", ascending=False))

# ---------------------------------------------------------------
# TASK 129: Departments with highest no-show percentage
# ---------------------------------------------------------------
print("\nTASK 129: NO-SHOW % BY DEPARTMENT")
df["Is_NoShow"] = (df["Attendance_Status"] == "No-show").astype(int)
noshow_dept = df.groupby("Department").agg(
    Appointments=("Appointment_ID", "count"),
    No_Shows=("Is_NoShow", "sum"))
noshow_dept["NoShow_Pct"] = (noshow_dept["No_Shows"] /
                             noshow_dept["Appointments"] * 100).round(2)
print(noshow_dept.sort_values("NoShow_Pct", ascending=False))

# ---------------------------------------------------------------
# TASK 130: No-show rate for SMS reminder Yes and No
# ---------------------------------------------------------------
print("\nTASK 130: SMS REMINDER VS ATTENDANCE")
print(pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"]))
print(pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"],
                  normalize="index").mul(100).round(2))
print("No-show % by SMS reminder:")
print((df.groupby("SMS_Reminder")["Is_NoShow"].mean() * 100).round(2))

# ---------------------------------------------------------------
# TASK 131: Average consultation fee by department
# ---------------------------------------------------------------
print("\nTASK 131: AVERAGE FEE BY DEPARTMENT")
print(df.groupby("Department")["Consultation_Fee"].mean()
      .round(2).sort_values(ascending=False))

# ---------------------------------------------------------------
# TASK 132: Appointment volume by date and department
# ---------------------------------------------------------------
print("\nTASK 132: VOLUME BY DATE AND DEPARTMENT")
volume = pd.crosstab(df["Appointment_Date"], df["Department"])
volume["Total"] = volume.sum(axis=1)
print(volume.head(10))
print("Busiest 5 dates:")
print(volume["Total"].sort_values(ascending=False).head(5))

df["Month"] = df["Appointment_Date"].dt.to_period("M")
print("Monthly volume by department:")
print(pd.crosstab(df["Month"], df["Department"]))

# ---------------------------------------------------------------
# TASK 133: Waiting times for new and follow-up appointments
# ---------------------------------------------------------------
print("\nTASK 133: WAIT - NEW VS FOLLOW-UP")
print(df.groupby("Appointment_Type")["Wait_Time_Min"]
      .agg(["count", "mean", "median", "max"]).round(2))

# ---------------------------------------------------------------
# TASK 134: NumPy median, quartiles, standard deviation
# ---------------------------------------------------------------
print("\nTASK 134: NUMPY STATISTICS OF WAITING TIME")
wait = df["Wait_Time_Min"].to_numpy()
q1 = np.percentile(wait, 25)
q3 = np.percentile(wait, 75)
print("Median:", np.median(wait))
print("Q1:", q1)
print("Q3:", q3)
print("Std deviation:", round(np.std(wait, ddof=1), 2))

# ---------------------------------------------------------------
# TASK 135: Unusually long waiting times using IQR
# ---------------------------------------------------------------
print("\nTASK 135: OUTLIERS USING IQR")
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("IQR:", iqr, "| Lower fence:", lower, "| Upper fence:", upper)
outliers = df[df["Wait_Time_Min"] > upper]
print("Unusually long waits:", len(outliers))
print(outliers[["Appointment_ID", "Department", "Wait_Time_Min"]])

# ---------------------------------------------------------------
# TASK 136: Waiting-time categories
# ---------------------------------------------------------------
print("\nTASK 136: WAITING-TIME CATEGORIES")
df["Wait_Band"] = pd.cut(df["Wait_Time_Min"],
                         bins=[0, 15, 30, 45, np.inf],
                         labels=["Short (0-15)", "Moderate (16-30)",
                                 "Long (31-45)", "Very long (over 45)"])
band = df["Wait_Band"].value_counts().sort_index()
print(pd.DataFrame({"Count": band,
                    "Percent": (band / len(df) * 100).round(2)}))

# ---------------------------------------------------------------
# TASK 137: Estimated total fees for attended appointments
# ---------------------------------------------------------------
print("\nTASK 137: ESTIMATED FEES (ATTENDED ONLY)")
attended = df[df["Attendance_Status"] == "Attended"]
print("Attended appointments:", len(attended))
print("Estimated total fees:", attended["Consultation_Fee"].sum())
print(attended.groupby("Department")["Consultation_Fee"].sum()
      .sort_values(ascending=False))

# ---------------------------------------------------------------
# TASK 138: Five operational recommendations
# ---------------------------------------------------------------
print("\nTASK 138: RECOMMENDATIONS")
print("1. Add extra reminders / confirmation calls in Pediatrics, Dermatology")
print("   and Orthopedics, which have the highest no-show rates.")
print("2. Keep sending SMS reminders; the no-show rate is lower with them.")
print("3. Review Pediatrics and Dermatology schedules to cut long waits.")
print("4. Investigate the very long waits (Orthopedics) and set a wait alert.")
print("5. Overbook slightly in high no-show departments and track weekly.")
print("(Operational view only - no medical conclusions.)")
