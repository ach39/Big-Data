import logging
from collections import namedtuple
from pathlib import Path
import sys
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.RandomState(1982)

def log(level, *args):
    for arg in args:
        logger.log(level, f"{arg}: {repr(eval(arg))}")

def info(*args): log(logging.INFO, *args)
def debug(*args): log(logging.DEBUG, *args)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
PROCESSED = "processed"
SEQS = "seqs"
LABELS = "labels"
IDS = "ids"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(f"{ROOT_DIR}/data/")
OUTPUT_DIR = Path(f"{ROOT_DIR}/output/")
MODEL_DIR = Path(f"{OUTPUT_DIR}")
FIG_DIR = Path(f"{ROOT_DIR}/doc/figs/")
LOG_DIR = Path(f"{OUTPUT_DIR}/logs/")

PATH_TRAIN = DATA_DIR / TRAIN
PATH_VALIDATION = DATA_DIR / VALIDATION
PATH_TEST = DATA_DIR / TEST
PATH_OUTPUT = OUTPUT_DIR / PROCESSED

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = PATH_OUTPUT / f"{SEQS}.{TRAIN}/"
PATH_TRAIN_LABELS = PATH_OUTPUT / f"{LABELS}.{TRAIN}/"
PATH_TRAIN_IDS = PATH_OUTPUT / f"{IDS}.{TRAIN}/"

PATH_VALID_SEQS = PATH_OUTPUT / f"{SEQS}.{VALIDATION}/"
PATH_VALID_LABELS = PATH_OUTPUT / f"{LABELS}.{VALIDATION}/"
PATH_VALID_IDS = PATH_OUTPUT / f"{IDS}.{VALIDATION}/"

PATH_TEST_SEQS = PATH_OUTPUT / f"{SEQS}.{TEST}/"
PATH_TEST_LABELS = PATH_OUTPUT / f"{LABELS}.{TEST}/"
PATH_TEST_IDS = PATH_OUTPUT / f"{IDS}.{TEST}/"

MODEL_PATH = MODEL_DIR / "best_rnn.path"
BASELINE_PATH = MODEL_DIR / "baseline_rnn.path"

SPLIT = namedtuple('TrainTestSplit', 'train validation test')(train=.8, validation=.15, test=.5)

info("PATH_OUTPUT", "ROOT_DIR", "DATA_DIR", "OUTPUT_DIR", "MODEL_DIR", "FIG_DIR", "SPLIT")

logging.basicConfig(filename=f"{LOG_DIR}/sepsis.log", level=logging.DEBUG)

