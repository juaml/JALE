import pandas as pd
import sys

def read_analysis_info(filepath):
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        sys.exit()
    except ValueError:
        print(f"Error reading Excel file '{filepath}'. Make sure it's a valid Excel file.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit()