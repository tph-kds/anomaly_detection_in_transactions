from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

CONFIG_FILE_PATH = Path("src/config_params/params.yaml")

ROOT_PROJECT = Path(os.getcwd())

