import pandas as pd

file_path = 'Unemployment in India.csv'
try:
    df = pd.read_csv(file_path)
    print("      UNEMPLOYMENT DATA CLEANING & REFINEMENT PIPELINE\n   ")
except FileNotFoundError:
    print(f"Error: Target file '{file_path}' not found in current workspace directory.")
    exit()

# 2. STANDARDIZE HEADERS & REMOVE EMPTY SPACER LINES
df.columns = df.columns.str.strip()

# Drop rows that are completely blank/empty lines (all values are NaN)
data_only = df.dropna(how='all').copy()


# 3. TRULY IDENTICAL DUPLICATES
print("[STEP 1] Checking row duplicates across ALL columns...")

all_column_duplicates = data_only[data_only.duplicated(keep=False)]

if len(all_column_duplicates) > 0:
    print(f" -> Found {data_only.duplicated().sum()} rows with identical column signatures.")
    print(" -> Displaying identified duplicate matching rows:")
    print(all_column_duplicates.sort_values(by=['Region', 'Date']))
    
    # Remove duplicates, keeping the first unique occurrence
    cleaned_df = data_only.drop_duplicates(keep='first').copy()
    print(" -> Duplicate matching rows safely removed.")
else:
    cleaned_df = data_only.copy()
    print(" -> Confirmed: Zero data rows contain 100% identical values across all columns.")


# 4. STRIP WHITESPACE ANOMALIES FROM ENTRIES
print("\n[STEP 2] Standardizing and stripping text value paddings...")
for col in cleaned_df.columns:
    if cleaned_df[col].dtype == 'object':
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()


# 5. TEMPORARY CHRONOLOGICAL SEQUENCE GROUPING BY STATE
print("\n[STEP 3] Structuring time variables and grouping by State...")
cleaned_df['Parsed_Date'] = pd.to_datetime(cleaned_df['Date'], format='%d-%m-%Y')

# Sort data rows by State (Region) first, and then sequentially by parsed ascending dates
cleaned_df = cleaned_df.sort_values(by=['Region', 'Parsed_Date'], ascending=[True, True])


# 6. FEATURE ENGINEERING: COMPUTE METRIC HEADCOUNTS & RATIOS
print("\n[STEP 4] Executing advanced data engineering and calculations...")

# Feature 1: Estimated Employment Rate (%)
cleaned_df['Estimated Employment Rate (%)'] = 100 - cleaned_df['Estimated Unemployment Rate (%)']

# Feature 2: Reverse-engineered Labor Force Headcount (Total Active Workforce)
cleaned_df['Estimated Labor Force'] = cleaned_df['Estimated Employed'] / (1 - (cleaned_df['Estimated Unemployment Rate (%)'] / 100))

# Feature 3: Isolate Unemployed Citizen Headcount
cleaned_df['Estimated Unemployed'] = cleaned_df['Estimated Labor Force'] - cleaned_df['Estimated Employed']

# Feature 4: Estimated Working-Age Population Pool Proxy
cleaned_df['Estimated Working Population Proxy'] = (cleaned_df['Estimated Labor Force'] / cleaned_df['Estimated Labour Participation Rate (%)']) * 100

# Feature 5 & 6: Granular Calendar Segment Isolation
cleaned_df['Year'] = cleaned_df['Parsed_Date'].dt.year
cleaned_df['Month_Name'] = cleaned_df['Parsed_Date'].dt.strftime('%B')

# Feature 7: COVID-19 Major Lockdown Flag (April 2020 and onward = 1, otherwise = 0)
cleaned_df['Is_Lockdown_Period'] = cleaned_df['Parsed_Date'].apply(lambda x: 1 if x >= pd.Timestamp('2020-04-01') else 0)


# Round large computed float headcounts to clean integers for visualization software
round_cols = ['Estimated Labor Force', 'Estimated Unemployed', 'Estimated Working Population Proxy']
cleaned_df[round_cols] = cleaned_df[round_cols].round(0).astype(int)

# Drop the temporary parsed date column to preserve native structure layout
final_dataset = cleaned_df.drop(columns=['Parsed_Date'])


output_filename = 'Final_Cleaned_Refined_Unemployment_India.csv'
final_dataset.to_csv(output_filename, index=False)

print("\n=========================================================")
print(f" PIPELINE COMPLETELY EXECUTE! Master file successfully saved. ")
print(f" Filename: '{output_filename}'")
print(f" Total Structured Data Rows Available: {len(final_dataset)} ")
print("=========================================================")