
import json
import csv

import os
# ============================================================
# JSON to CSV Converter
# Handles multiple JSON objects, invalid formats, and extra data
# ============================================================
# === Step 1: Specify the input JSON file path ===
input_json_path = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0.json"
# Automatically generate output CSV path with the same name
output_csv_path = os.path.splitext(input_json_path)[0] + ".csv"
# === Step 2: Function to safely load JSON data ===
def load_json_data(file_path):
    """
    Safely loads JSON data from a file.
    Handles cases like:
      - Multiple JSON objects (not inside an array)
      - Extra data after valid JSON
      - Invalid JSON formatting
    Returns a list of dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        # Try parsing as a full JSON structure (array or object)
        data = json.loads(content)
        # Ensure it's a list for CSV conversion
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("Unsupported JSON structure for CSV conversion.")
        return data
    except json.JSONDecodeError as e:
        # Handle "Extra data" or multiple JSON objects
        if "Extra data" in str(e):
            print("⚠ Warning: Multiple JSON objects detected. Attempting line-by-line parsing...")
            # Try JSON Lines (each line is a valid JSON object)
            data = []
            for line_num, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as err_line:
                    print(f"❌ Skipping invalid JSON at line {line_num}: {err_line}")
            if not data:
                raise ValueError("No valid JSON objects found in the file.")
            return data
        else:
            # General invalid JSON error
            raise ValueError(f"Failed to parse JSON: {e}")
# === Step 3: Function to write data to CSV ===
def write_csv(data, output_path):
    """
    Converts a list of JSON objects (dicts) to a CSV file.
    Handles missing or extra keys gracefully.
    """
    # Collect all unique keys across all JSON objects
    all_keys = set()
    for item in data:
        if isinstance(item, dict):
            all_keys.update(item.keys())
        else:
            raise ValueError("Each JSON element must be an object (dictionary).")
    # Write data to CSV file
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Conversion successful! CSV file saved at: {output_path}")
# === Step 4: Main execution ===
if __name__ == "__main__":
    try:
        json_data = load_json_data(input_json_path)
        write_csv(json_data, output_csv_path)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")


