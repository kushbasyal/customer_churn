import os
import json
# Project root directory (go up one level from scripts folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to config.json
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# Load config.json
with open(CONFIG_FILE, "r") as file:
    config = json.load(file)

# Extract paths from config.json
RAW_DATA_PATH = os.path.join(BASE_DIR, config["paths"]["raw_data"])
CLEANED_DATA_PATH = os.path.join(BASE_DIR, config["paths"]["cleaned_data"])

# Optional extra paths (for future use)
ANALYSIS_NOTEBOOK_PATH = os.path.join(BASE_DIR, config["paths"]["analysis_notebook"])
MAIN_NOTEBOOK_PATH = os.path.join(BASE_DIR, config["paths"]["main_notebook"])
SQL_FILE_PATH = os.path.join(BASE_DIR, config["paths"]["sql_file"])
POWERBI_DASHBOARD_PATH = os.path.join(BASE_DIR, config["paths"]["powerbi_dashboard"])
ANALYSIS_DOC_PATH = os.path.join(BASE_DIR, config["paths"]["analysis_doc"])
MODELING_DOC_PATH = os.path.join(BASE_DIR, config["paths"]["modeling_doc"])
MODELING_NOTEBOOK_PATH = os.path.join(BASE_DIR, config["paths"]["modeling_notebook"])
CLUSTERING_NOTEBOOK_PATH = os.path.join(BASE_DIR, config["paths"]["clustering_notebook"])
