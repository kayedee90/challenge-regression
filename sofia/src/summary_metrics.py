import os
import pandas as pd

# Define paths
reports_folder = r"C:\Users\sofia\BeCode\Projects\ImmoEliza\03 challenge-regression\sofia\outputs\reports"
output_csv = r"C:\Users\sofia\BeCode\Projects\ImmoEliza\03 challenge-regression\sofia\outputs\summary_reports.csv"

# List the available report files
def list_reports(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Read report files into a dictionary
def read_reports(report_paths):
    reports_content = {}
    for path in report_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            content = content.replace('²', '2')  # Replace problematic characters
        reports_content[path] = content
    return reports_content

# Parse metrics from report content
def parse_report(content):
    metrics = {}
    for line in content.splitlines():
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                # ignore lines where value can't convert to float
                pass
    return metrics

  

# Save the metrics to a CSV
def save_reports_table(reports_content, output_path):
    rows = []
    for path, content in reports_content.items():
        metrics = parse_report(content)   # Now this will return a flat dict with Train_R2, Test_RMSE, etc.
        metrics['filename'] = os.path.basename(path)
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Summary saved to: {output_path}")

    
# Run the workflow
report_paths = list_reports(reports_folder)
reports_content = read_reports(report_paths)
save_reports_table(reports_content, output_csv)
