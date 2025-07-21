import os
import pandas as pd
import sys

# Check if directory is passed
if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

# Get input directory path from command-line argument
input_dir = sys.argv[1]

# Prepare results
results = []

# File ID counter
file_id = 1

# Traverse all result.csv files
for filename in os.listdir(input_dir):
    if filename.endswith("result.csv"):
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path)  # CSV is comma-separated
            if 'Probability' in df.columns:
                avg_prob = df['Probability'].mean()
                prediction = 1 if avg_prob > 0.5 else 0
                results.append({
                    "ID": f"File_{file_id}",
                    "Average_Probability": avg_prob,
                    "Prediction": prediction
                })
                file_id += 1
            else:
                print(f"Warning: 'Probability' column not found in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Save output
output_file = "summary_predictions.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)

# Summary
num_positive = (output_df["Prediction"] == 1).sum()
print(f"Results saved to: {output_file}")
print(f"Number of predictions with value 1: {num_positive}")