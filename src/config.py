from pathlib import Path

COMP_NAME = "icecube-neutrinos-in-deep-ice"
STORAGE = "storage_dimm2"
# STORAGE = "storage"

INPUT_PATH = Path(f"/mnt/{STORAGE}/kaggle_data/{COMP_NAME}/")
INPUT_PATH_ALT = Path(f"/mnt/storage/kaggle_data/{COMP_NAME}/")
OUTPUT_PATH = Path(f"/mnt/{STORAGE}/kaggle_output/{COMP_NAME}/")
CONFIG_PATH = Path(f"/home/anjum/kaggle/{COMP_NAME}/hyperparams.yml")
MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
# MODEL_CACHE = Path("/mnt/storage/model_cache/huggingface")
