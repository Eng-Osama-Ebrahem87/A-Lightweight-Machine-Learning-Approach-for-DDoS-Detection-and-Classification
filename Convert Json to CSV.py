import json
import csv

import os

from typing import List, Dict, Any, Union
def json_to_csv_converter(input_json_path: str, output_csv_path: str = None) -> bool:
    """
    Convert JSON file to CSV file with comprehensive error handling
    and support for various JSON formats including multiple JSON objects.
    Args:
        input_json_path (str): Path to the input JSON file
        output_csv_path (str, optional): Path for the output CSV file. 
                                       If None, uses input path with .csv extension
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Validate input file existence
    if not os.path.exists(input_json_path):
        print(f"Error: Input file '{input_json_path}' does not exist.")
        return False
    # Generate output path if not provided
    if output_csv_path is None:
        base_name = os.path.splitext(input_json_path)[0]
        output_csv_path = f"{base_name}.csv"
    try:
        # Read the entire file content
        with open(input_json_path, 'r', encoding='utf-8') as json_file:
            file_content = json_file.read().strip()
        # Try to parse as standard JSON first
        try:
            parsed_data = json.loads(file_content)
            return process_json_data(parsed_data, output_csv_path)
        except json.JSONDecodeError as e:
            print(f"Standard JSON parsing failed: {e}")
            print("Attempting alternative parsing methods...")
            # Try to handle multiple JSON objects
            return handle_multiple_json_objects(file_content, output_csv_path)
    except Exception as e:
        print(f"Unexpected error during file processing: {e}")
        return False
def process_json_data(data: Any, output_csv_path: str) -> bool:
    """
    Process parsed JSON data and convert to CSV format.
    Args:
        data: Parsed JSON data (dict, list, or other)
        output_csv_path (str): Path for output CSV file
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Handle different JSON structures
        if isinstance(data, dict):
            # Single JSON object
            return write_dict_to_csv([data], output_csv_path)
        elif isinstance(data, list):
            # Array of JSON objects
            if all(isinstance(item, dict) for item in data):
                return write_dict_to_csv(data, output_csv_path)
            else:
                print("Error: JSON array contains non-object elements")
                return False
        else:
            print(f"Error: Unsupported JSON structure type: {type(data)}")
            return False
    except Exception as e:
        print(f"Error processing JSON data: {e}")
        return False
def handle_multiple_json_objects(file_content: str, output_csv_path: str) -> bool:
    """
    Handle files containing multiple JSON objects or JSON Lines format.
    Args:
        file_content (str): Raw file content
        output_csv_path (str): Path for output CSV file
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Try JSON Lines format (one JSON object per line)
        lines = file_content.split('\n')
        json_objects = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    json_objects.append(obj)
                else:
                    print(f"Warning: Line {line_num} contains non-object JSON")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
                continue
        if json_objects:
            return write_dict_to_csv(json_objects, output_csv_path)
        # Try to extract multiple JSON objects from continuous string
        return extract_multiple_json_objects(file_content, output_csv_path)
    except Exception as e:
        print(f"Error handling multiple JSON objects: {e}")
        return False
def extract_multiple_json_objects(file_content: str, output_csv_path: str) -> bool:
    """
    Extract multiple JSON objects from a continuous string.
    Args:
        file_content (str): Raw file content
        output_csv_path (str): Path for output CSV file
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        json_objects = []
        start_pos = 0
        content_length = len(file_content)
        while start_pos < content_length:
            # Find the next opening brace
            brace_pos = file_content.find('{', start_pos)
            if brace_pos == -1:
                break
            # Try to parse from this position
            try:
                # Parse JSON starting from the opening brace
                obj, end_pos = parse_json_from_position(file_content, brace_pos)
                if obj is not None:
                    json_objects.append(obj)
                    start_pos = end_pos
                else:
                    start_pos = brace_pos + 1
            except:
                start_pos = brace_pos + 1
        if json_objects:
            return write_dict_to_csv(json_objects, output_csv_path)
        else:
            print("Error: Could not extract any valid JSON objects")
            return False
    except Exception as e:
        print(f"Error extracting multiple JSON objects: {e}")
        return False
def parse_json_from_position(content: str, start_pos: int) -> tuple:
    """
    Parse JSON object starting from a specific position in the string.
    Args:
        content (str): The complete content string
        start_pos (int): Starting position to begin parsing
    Returns:
        tuple: (parsed_object, end_position) or (None, start_pos+1) if failed
    """
    try:
        # Use JSON decoder to find the end of the JSON object
        decoder = json.JSONDecoder()
        obj, end = decoder.raw_decode(content, start_pos)
        return obj, end
    except json.JSONDecodeError:
        return None, start_pos + 1
def write_dict_to_csv(data: List[Dict], output_csv_path: str) -> bool:
    """
    Write list of dictionaries to CSV file.
    Args:
        data (List[Dict]): List of dictionary objects to write
        output_csv_path (str): Path for output CSV file
    Returns:
        bool: True if successful, False otherwise
    """
    if not data:
        print("Error: No data to write")
        return False
    try:
        # Collect all unique fieldnames from all objects
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(list(fieldnames))
        # Write to CSV
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                # Ensure all fields are present in each row
                complete_row = {field: row.get(field, '') for field in fieldnames}
                writer.writerow(complete_row)
        print(f"Successfully converted JSON to CSV: {output_csv_path}")
        print(f"Processed {len(data)} records with {len(fieldnames)} columns")
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False
def main():
    """
    Main function to demonstrate the JSON to CSV converter.
    """
    # Specify your input JSON file path here
    input_json_path = r"E:\VeReMi Dataset\Some VeReMi files\traceJSON-9-7-A0-1-0.json"
    # Convert JSON to CSV
    success = json_to_csv_converter(input_json_path)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
if __name__ == "__main__":
    main()
'''
This professional Python code provides a comprehensive JSON to CSV converter with the following features:
Key Features:
1. Multiple JSON Format Support:
   · Standard JSON objects/arrays
   · JSON Lines format (one JSON per line)
   · Multiple JSON objects in single file
   · Handles extra characters and whitespace
2. Comprehensive Error Handling:
   · File existence validation
   · JSON parsing errors recovery
   · Graceful handling of malformed data
3. Flexible Input Processing:
   · Single JSON objects
   · Arrays of JSON objects
   · Multiple independent JSON objects
   · JSON Lines format
4. Automatic Output Generation:
   · Generates CSV path from JSON path if not specified
   · Collects all unique fieldnames automatically
   · Handles missing fields in objects
Usage:
Simply change the input_json_path variable to point to your JSON file:
```python
input_json_path = "path/to/your/file.json"
```
The code will automatically handle various JSON formats and generate a corresponding CSV file with all data preserved.

'''
