from util.config import *
from torchtext.data import Field
import pandas as pd
from util.utils import get_train_test_split
from torchtext.data import TabularDataset
logger

if __name__ == "__main__":
    # https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
    df = pd.read_csv(DATA_DIR / 'text_features_1575527882.csv')
    logger.info(f"Text records {df.shape[0]}")
    logger.info(f"Text records {df['SUBJECT_ID'].unique().shape[0]}")
    logger.info(f"Text records {df['SUBJECT_ID'].value_counts()}")
    train, valid, test = get_train_test_split(df['SUBJECT_ID'].unique().tolist())


    print(df.head())
    TEXT = Field(sequential=True,
                 tokenize=lambda x: x.split())
    LABEL = Field(sequential=False,
                  use_vocab=False)

    nlp_datafields = [('SUBJECT_ID', None),
                      ('HADM_ID', None),
                      ('ROW_ID', None),
                      ('CHARTDATE', None),
                      ('CHARTTIME', None),
                      ('STORETIME', None),
                      ('CATEGORY', None),
                      ('DESCRIPTION', None),
                      ('CGID', None),
                      ('ISERROR', None),
                      ('TEXT', TEXT),
                      ('dt_min', None),
                      ('dt_max', None),
                      ('label', LABEL)]
