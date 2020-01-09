import pickle

import pandas as pd
from sklearn import preprocessing

from util.config import *

logger = logging.getLogger(__name__)

def import_csv(path=DATA_DIR / 'data20191125.csv',
               index=['subject_id', 'hadm_id', 'index_hrs']):
    csv = pd.read_csv(path, index_col=index)

    stats = {'num_records': csv.shape[0],
             'num_features': csv.shape[1],
             'case': csv.loc[csv['CaseControl'] == 'Case'].shape[0],
             'control': csv.loc[csv['CaseControl'] == 'Control'].shape[0],
             'other': csv.loc[~csv['CaseControl'].isin(['Control', 'Case'])].shape[0]
             }
    logger.info(f"Importing {path}, indexing by {index}")
    logger.info(csv.columns)
    logger.info(stats)

    logger.debug(csv.shape)
    return csv


def preprocess_data(df):
    # remove observation data
    logger.debug(df.shape)
    index_names = df.index.names
    df = df.reset_index()

    base_count = df.shape[0]
    df = df.loc[df['Pred or Obs'] == 'Prediction']
    logger.info("removed {} Prediction record(s)".format(base_count - df.shape[0]))
    base_count = df.shape[0]

    # remove any data before 12 hours
    df = df.loc[df['index_hrs'] <= 12]
    logger.info("removed {} > 12 hr record(s)".format(base_count - df.shape[0]))
    base_count = df.shape[0]

    # remove anyone younger than 15
    df = df.loc[df['admission_age'] > 15]
    logger.info("removed {} record(s) under age 16".format(base_count - df.shape[0]))

    logger.debug(df.shape)
    if index_names[0] is not None:
        df = df.set_index(index_names)
    logger.debug(df.shape)
    return df


def get_case_control(df):
    logger.debug(f'df size {df.shape}, columns: {df.columns}')
    case = df.loc[df['CaseControl'] == 'Case', 'subject_id'].unique()
    control = df.loc[df['CaseControl'] == 'Control', 'subject_id'].unique()
    logger.info('{} Case patients, {} Control patients'.format(case.size, control.size))
    return case, control


def downsample_control(df):
    index_names = df.index.names
    df = df.reset_index()
    case, control = get_case_control(df)
    index = np.random.choice(control.size, case.size, replace=False)
    control = control[index]
    all = np.concatenate([case, control])
    df = df.loc[df['subject_id'].isin(all)]
    _, _ = get_case_control(df)  # Just to log the new counts
    df = df.set_index(index_names)
    return df


def create_labels(csv):
    labels_ds = csv.reset_index()
    labels_ds['label'] = 0
    labels_ds.loc[labels_ds['CaseControl'] == 'Case', 'label'] = 1
    labels_ds = labels_ds.groupby('subject_id').first()['label']
    logger.info("value_counts: {}".format(labels_ds.value_counts()))
    return labels_ds


def fill_na(df_tmp):
    # df_tmp = df_tmp.sort_values(by='index_hrs', ascending=False)
    df_tmp = df_tmp.fillna(method='ffill')
    df_tmp = df_tmp.fillna(method='bfill')
    return df_tmp


def handle_missing_values(df):
    df = df.groupby(['subject_id', 'hadm_id']).apply(fill_na)
    df = df.fillna(df.mean())
    df = df.fillna(0)
    assert(not df.isnull().values.any())
    return df


def create_dataset(df, features, labels_ds, scaler=preprocessing.MinMaxScaler()):
    logger.info("creating dataset with features {}".format(features))
    df = df.sort_index(ascending=[True, True, False])
    df = handle_missing_values(df)

    # fit scaler normalize feature values [0, 1.0]
    scaler = scaler.fit(df[features])

    patients = []
    labels = []
    charts = []

    # Loop over the groups, create a list of patient_ids, labels, and list of list of visits
    for (patient, hadm), df_tmp in df.groupby(['subject_id', 'hadm_id']):
        assert (df_tmp.shape[1] == len(features))

        try:
            labels.append(labels_ds.loc[patient])
            charts.append(scaler.transform(df_tmp).tolist())
            patients.append(patient)
        except KeyError:
            logger.exception("{} not found in labels_ds".format(patient))

    assert (len(patients) == len(labels))
    assert (len(charts) == len(labels))

    logger.info("created dataset for {} patients".format(len(patients)))
    return patients, labels, charts


def split_patients(df):
    pool = pd.Series(df.index.get_level_values('subject_id'))
    train = pool.sample(frac=SPLIT.train)
    validation = pool.sample(frac=SPLIT.validation)
    test = pool.loc[np.setdiff1d(train.array, validation.array)]

    logger.info(f"train={train.count()} val={validation.count()} test={test.count()}")
    train_df = df.iloc[df.index.get_level_values('subject_id').isin(train)]
    val_df = df.iloc[df.index.get_level_values('subject_id').isin(validation)]
    test_df = df.iloc[df.index.get_level_values('subject_id').isin(test)]

    num_test = test.shape[0]
    num_validation = validation.shape[0]
    num_train = train.shape[0]
    total = num_test + num_train + num_validation
    
    logger.info(f"Train: {num_train / total}, Val: {num_validation / total}, Test, "
                f"{num_train / total} Train, {total} Records")
    
    return train_df, val_df, test_df


def main():
    features = ['admission_age', 'avg_HeartRate', 'avg_SysBP', 'avg_DiasBP', 'avg_MeanBP',
                'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose',
                'AVG HEARTRATE','AVG SYSBP','AVG DIASBP','AVG MEANBP','AVG RESPRATE','AVG TEMPC',
                'AVG SPO2','AVG GLUCOSE','GCS','GCSMotor','GCSVerbal','GCSEyes','EndoTrachFlag',
                'SPECIMEN_PROB','SO2','spo2','PO2','PCO2','fio2_chartevents','FIO2','AADO2','AADO2_calc',
                'PaO2FiO2Ratio','PH','BASEEXCESS','TOTALCO2','CARBOXYHEMOGLOBIN','METHEMOGLOBIN','CALCIUM',
                'TEMPERATURE','INTUBATED','TIDALVOLUME','VENTILATIONRATE','VENTILATOR','PEEP','O2Flow','REQUIREDO2',
                'ANIONGAP','ALBUMIN','BANDS','BICARBONATE','BILIRUBIN','CREATININE','CHLORIDE','GLUCOSE','HEMATOCRIT',
                'HEMOGLOBIN','LACTATE','PLATELET','POTASSIUM','PTT','INR','PT','SODIUM','BUN','WBC']

    csv = import_csv()
    labels_ds = create_labels(csv)

    processed = preprocess_data(csv)
    downsampled = downsample_control(processed)
    train, validation, test = split_patients(downsampled[features])

    train_ids, train_labels, train_seqs = create_dataset(train, features, labels_ds)
    validation_ids, validation_labels, validation_seqs = create_dataset(validation, features, labels_ds)
    test_ids, test_labels, test_seqs = create_dataset(test, features, labels_ds)
    
    def dump(data, path):
        pickle.dump(data, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)

    # Train set
    logger.info("Construct train set")
    dump(train_ids, PATH_TRAIN_IDS)
    dump(train_labels, PATH_TRAIN_LABELS)
    dump(train_seqs, PATH_TRAIN_SEQS)

    # # Validation set
    logger.info("Construct validation set")
    dump(validation_ids, PATH_VALID_IDS)
    dump(validation_labels, PATH_VALID_LABELS)
    dump(validation_seqs, PATH_VALID_SEQS)

    # Test set
    logger.info("Construct test set")
    dump(test_ids, PATH_TEST_IDS)
    dump(test_labels, PATH_TEST_LABELS)
    dump(test_seqs, PATH_TEST_SEQS)

    logger.info("Complete!")


if __name__ == '__main__':
    main()
