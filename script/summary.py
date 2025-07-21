import os
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

input_dir = sys.argv[1]

all_data = []

for filename in os.listdir(input_dir):
    if filename.endswith("result.csv"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path)
            if {'ID', 'Prediction', 'Probability'}.issubset(df.columns):
                all_data.append(df[['ID', 'Prediction', 'Probability']])
            else:
                print(f"Warning: Columns missing in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if not all_data:
    print("No valid result.csv files found.")
    sys.exit(1)

combined_df = pd.concat(all_data, ignore_index=True)

summary_df = combined_df.groupby('ID').agg({
    'Prediction': 'mean',
    'Probability': 'mean'
}).reset_index()

summary_df['Prediction'] = summary_df['Prediction'].apply(lambda x: 1 if x > 0.5 else 0)

output_file = "summary_predictions.csv"
summary_df.to_csv(output_file, index=False)

num_positive = (summary_df["Prediction"] == 1).sum()
print(f"Results saved to: {output_file}")
print(f"Number of predictions with value 1: {num_positive}")