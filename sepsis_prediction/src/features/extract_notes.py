from util.config import *
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
import logging
import re

nltk.download('stopwords')
nltk.download('punkt')

logger = logging.getLogger(__name__)
stop_words = set(stopwords.words('english'))

def open_csv(filename="NOTEEVENTS.csv", index=['SUBJECT_ID', 'HADM_ID']):
    """
    Opens the Mimiciii notevents CSV file from the data folder
    :param filename: the name of the note events csv file
    :return: a Dataframe of all the note events.
    """
    df = pd.read_csv(DATA_DIR / filename,
                     index_col=index,
                     # nrows=1000,
                     infer_datetime_format=True)
    logger.info(f"opening {filename}")
    logger.info(f"Dataframe columns: {df.columns}")
    # logger.info(f"Clinical note types: {df['CATEGORY'].unique()}")
    return df

def get_cohort_trimmed(filename='cohort_trimmed.csv'):
    df = open_csv(filename, index=['subject_id'])
    return df

def write_to_file(df):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    df.to_csv(DATA_DIR / f'text_features_{int(timestamp)}.csv')

def get_ids():
    train_ids = pickle.load(open(PATH_TRAIN_IDS, "rb"))
    validation_ids = pickle.load(open(PATH_VALID_IDS, "rb"))
    test_ids = pickle.load(open(PATH_TEST_IDS, "rb"))
    logger.info(train_ids.extend(validation_ids))
    all_ids = []
    all_ids.extend(train_ids)
    all_ids.extend(validation_ids)
    all_ids.extend(test_ids)
    all_ids = set(all_ids)

    logger.info(f"Loaded {len(train_ids)} train ids, {len(validation_ids)} "
                f"validation ids, {len(test_ids)} test ids, "
                f"and {len(all_ids)} all ids.")
    return train_ids, validation_ids, test_ids, all_ids

def process_text(text, stemmer=SnowballStemmer("english"), min_length=3):
    """
    Remove stop words and other unprintable characters
    :param text: A multline clinical note
    :return: an array of tokens
    """
    text = text.lower()
    text = re.sub('dictated.*', '', text, flags=re.MULTILINE|re.DOTALL)
    text = re.sub('.*:\s+', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\s\s+', ' ', text)
    text = re.sub('[,.]', '', text)
    text = re.sub('[/-]', ' ', text)
    tokens = word_tokenize(text)
    return " ".join([stemmer.stem(t) for t in tokens if t not in stop_words
                     and len(t) >= min_length])

if __name__ == "__main__":
    df = open_csv()
    logger.info(f"{df.shape[0]} chart records loaded")
    cohort = get_cohort_trimmed()
    logger.info(f"{cohort.shape[0]} cohort patients indexed")
    cohort_indx = cohort.index.values

    filtered = df.loc[cohort_indx]
    logger.info(f"{filtered.shape[0]} cohort charts selected")
    joined = filtered.join(cohort, on='SUBJECT_ID')
    observation_charts = joined.loc[(joined['CHARTDATE'] > joined['dt_min']) & (joined['CHARTDATE'] < joined['dt_max'])]
    logger.info(f"{filtered.shape[0]} observation_charts")

    # # remove unprintable characters, stop words and short words
    observation_charts['TEXT']=observation_charts['TEXT'].map(process_text)

    # # write text to file
    write_to_file(observation_charts)
