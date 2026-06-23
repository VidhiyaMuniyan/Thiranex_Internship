# ============================================================
# DATA CLEANING & REPORT AUTOMATION PROJECT
# ============================================================

# ============================================================
# STEP 1 : IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime


# ============================================================
# STEP 2 : LOAD DATA
# ============================================================

df = pd.read_csv("sales_data.csv")

print("Original Shape:", df.shape)

print(df.head())


# ============================================================
# STEP 3 : DATA CLEANING
# ============================================================

print("\nMissing Values Before Cleaning")

print(df.isnull().sum())


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates")

print(df.shape)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    df[col].fillna(df[col].median(), inplace=True)


categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:

    df[col].fillna(df[col].mode()[0], inplace=True)


# ============================================================
# REMOVE EXTRA SPACES
# ============================================================

for col in categorical_columns:

    df[col] = df[col].str.strip()


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns After Cleaning")

print(df.columns)


# ============================================================
# CHECK FOR NEGATIVE SALES
# ============================================================

if "sales" in df.columns:

    df = df[df["sales"] >= 0]


# ============================================================
# CONVERT DATE COLUMN
# ============================================================

if "date" in df.columns:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


# ============================================================
# FINAL NULL CHECK
# ============================================================

print("\nMissing Values After Cleaning")

print(df.isnull().sum())


# ============================================================
# SAVE CLEANED DATA
# ============================================================

df.to_csv(
    "cleaned_sales_data.csv",
    index=False
)

print("\nCleaned Dataset Saved")


# ============================================================
# STEP 4 : KPI CALCULATIONS
# ============================================================

total_records = len(df)

total_sales = (
    df["sales"].sum()
    if "sales" in df.columns
    else 0
)

average_sales = (
    df["sales"].mean()
    if "sales" in df.columns
    else 0
)

maximum_sale = (
    df["sales"].max()
    if "sales" in df.columns
    else 0
)

minimum_sale = (
    df["sales"].min()
    if "sales" in df.columns
    else 0
)


# ============================================================
# KPI SUMMARY TABLE
# ============================================================

kpi_df = pd.DataFrame({

    "Metric":[
        "Total Records",
        "Total Sales",
        "Average Sales",
        "Maximum Sale",
        "Minimum Sale"
    ],

    "Value":[
        total_records,
        total_sales,
        average_sales,
        maximum_sale,
        minimum_sale
    ]
})

print("\nKPI Summary")

print(kpi_df)


# ============================================================
# STEP 5 : SALES BY CATEGORY
# ============================================================

if "category" in df.columns and "sales" in df.columns:

    category_sales = (
        df.groupby("category")["sales"]
        .sum()
        .reset_index()
    )

    print("\nCategory Sales")

    print(category_sales)


# ============================================================
# STEP 6 : MONTHLY SALES ANALYSIS
# ============================================================

if "date" in df.columns and "sales" in df.columns:

    df["month"] = df["date"].dt.month_name()

    monthly_sales = (
        df.groupby("month")["sales"]
        .sum()
        .reset_index()
    )

    print("\nMonthly Sales")

    print(monthly_sales)


# ============================================================
# STEP 7 : AUTOMATED VISUALIZATION
# ============================================================

if "category" in df.columns and "sales" in df.columns:

    plt.figure(figsize=(10,6))

    plt.bar(
        category_sales["category"],
        category_sales["sales"]
    )

    plt.title("Sales By Category")

    plt.xlabel("Category")

    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig("category_sales.png")

    plt.close()


# ============================================================
# MONTHLY SALES TREND
# ============================================================

if "date" in df.columns and "sales" in df.columns:

    monthly_trend = (
        df.groupby("date")["sales"]
        .sum()
    )

    plt.figure(figsize=(12,6))

    plt.plot(monthly_trend)

    plt.title("Sales Trend")

    plt.xlabel("Date")

    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig("sales_trend.png")

    plt.close()


# ============================================================
# STEP 8 : EXCEL REPORT AUTOMATION
# ============================================================

with pd.ExcelWriter(
    "Automated_Business_Report.xlsx"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Cleaned Data",
        index=False
    )

    kpi_df.to_excel(
        writer,
        sheet_name="KPI Summary",
        index=False
    )

    if "category" in df.columns:

        category_sales.to_excel(
            writer,
            sheet_name="Category Analysis",
            index=False
        )

    if "date" in df.columns:

        monthly_sales.to_excel(
            writer,
            sheet_name="Monthly Sales",
            index=False
        )

print("\nExcel Report Generated Successfully")


# ============================================================
# STEP 9 : TEXT REPORT AUTOMATION
# ============================================================

with open(
    "Business_Report.txt",
    "w"
) as report:

    report.write(
        "BUSINESS PERFORMANCE REPORT\n"
    )

    report.write(
        "="*50 + "\n\n"
    )

    report.write(
        f"Generated On: {datetime.now()}\n\n"
    )

    report.write(
        f"Total Records : {total_records}\n"
    )

    report.write(
        f"Total Sales : {total_sales}\n"
    )

    report.write(
        f"Average Sales : {average_sales:.2f}\n"
    )

    report.write(
        f"Maximum Sale : {maximum_sale}\n"
    )

    report.write(
        f"Minimum Sale : {minimum_sale}\n"
    )

print("\nBusiness Report Created")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\nData Cleaning & Report Automation Completed")
