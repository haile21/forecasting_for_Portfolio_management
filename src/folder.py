import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

# lsit of files we want them to be created
list_of_files = [
    ".vscode/settings.json",
    ".github/workflows/unittests.yml",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "src/__init__.py",
    "data/sample.csv",
    "notebooks/__init__.py",
    "notebooks/README.md",
    "scripts/__init__.py",
    "scripts/README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)  # changing to the actual path
    # trying to split the file name and directory name
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory; {filedir} for the file {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already created")