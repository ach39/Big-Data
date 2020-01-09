# Sepsis Prediction in ICU patients
### achauhan39 , Georgia Institute of Technology, GA 

-------------------------------------------------------------
## 1. Environment

1. Sepsis detection criteria and cohort construction was implemented using Big-Query on Google Cloud Platform.
	- Copy queries provide in code/Bigquery folder in [BigQuery editor](https://console.cloud.google.com/bigquery) and hit 'Run'.
	- In order to run the queries, access to MIMIC-III cloud data is required. Follow [cloud data access guide](https://mimic.physionet.org/tutorials/intro-to-mimic-iii-bq/) to request access.
2. Data preprocessing was done in pyspark.
3. Model development was done in python and pytorch.
	
```bash
> cd sepsis_prediction
> conda env create -f environment.yml
> conda activate sepsis_prediction
(sepsis_prediction)>

```


-------------------------------------------------------------
## 2. Directory structure

```bash
CSE6250 Group Project
sepsis_prediction
│   README.md
│   environment.yml   
│
└─── data
|   │- data20191125.csv      		    (preprocessed data, input to ML models)
|   │- cohort_trimmed.csv      		    (preprocessed data, input to ML models)
|   │- text_features_1575446631.csv     (preprocessed data, input to ML models)
|   |- NOTEEVENTS.csv                   mimiciii charts (Not included, obtain from physionet)
|
└─── doc
|   |
|   |- Team6_SepsisPrediction.pdf       (Research paper describing the methods and results)
|   |___ figs                           (Generated plots from train_variable_rnn.py)
|       
└─── output				(best models)
│   │- svm.pkl
│   │- logisticRegression.pkl
│   │- mlp.pkl
│   │- gradientBoost.pkl
│   │- rnn.path
|   |____ processed                     (Directory containing training data for LSTM)
|   |____ logs                          (Directory contining log files)
│
└─── src│   │
│   └─── big_query			(Big-Queries to data from MIMIC-III data hosted on Google Cloud)
│   │    │- sepis_onset_time.sql	(query to compute sepsis onset time)
│   │    │- pivoted_bg_art.sql          (query to get blood-gas measures)
|   |    |- pivoted_gcs.sql		(query to get neurologcial measures GCS)
|   |    |- pivoted_lab.sql		(query to get lab results)
|   |    |- vital_icustay_detail.sql	(query to get icustay details)
|   |
│   └─── features
│   │   │- etl_sepsis_data.py
│   │   │- etl_spark.py
│   │   │- extract_notes.py
│   │
│   └─── other_models
│   |   │- ac_models.py			(code to create and tune svm, mlp, logReg and gradientBoost model)
│   |   │- ac_utils.py
|   |
│   └─── rnn
│   |   │- lstm.py                      (Defines lstm model)
│   |   │- mydatasets.py                (Provides data to LSTM training algorithm)
│   |   │- train_variable_rnn.py        (Trains lstm model and generates plots)
|   |
│   └─── util
│       │- config.py                    (Configures input and output paths)
│       │- plots.py                     (Generates plots for LSTM model)
│       │- utils.py                     (Helper functions for LSTM model)
│
└─── tst
    │- test_etl_sepsis_data.py
    │- test_extract_notes.py
   
```

-------------------------------------------------------------
## 3. Data

Data from following MIMIC-III tables was extracted and processed to produce dataset for modelling.

```
 physionet-data.mimiciii_derived.icustay_detail
 physionet-data.mimiciii_derived.suspicion_of_infection
 physionet-data.mimiciii_derived.pivoted_sofa
 physionet-data.mimiciii_derived.pivoted_vital
 physionet-data.mimiciii_derived.pivoted_gcs
 physionet-data.mimiciii_derived.pivoted_lab
 physionet-data.mimiciii_derived.pivoted_bg_art
```


-------------------------------------------------------------
## 4. High level workflow

1.	**Identify case and control patients** : Sepsis onset time was determined using SOFA>= 2 criteria. Using this information, ICU patients (> 15 years)  were split into case and control group. 
2.	**Define observation and prediction window** : observation window was set to 12 hours and observation window was set to 6 hours. 
3.	**Data cleanup and Feature engineering** :  Our current research shows that vitals and age have high predictive power for this task and hence were  chosen as base feature set.
4.	**Feature reduction and model creation**
    1.	Following set of  supervised models were developed to compare and contrast predictive power of each ML-Algorithm for this dataset.
                o	SVM , Logistic Regression , XGBoost, MLP
    2.	Since MIMIC data is timestamped, we leveraged this information to train a RNN network.


          ![alt text](https://github.gatech.edu/tweldon7/sepsis_prediction/blob/master/doc/workflow.png)
 
 
#### Filtering Rules
    Case patients
        1. sepsis_onset_time – icu_in_time > = 18 hours and
        2. number of hourly records in observation window >= 3 and age > 15
    Control patients
        1.	Select patients that don't appear in any sepsis criteria (i.e. in sepsis_onset_time.csv or sepsis_superset_patientIDs.csv) and 
        2.	Patient's icu_out_time - icu_in_time > 18 hours and 
        3.	number of hourly records in observation window >= 3 and age > 15
    This resulted in 546 case patients. To keep dataset somewhat balanced, similar number of control patients were randomly selected after applying above rules.

-------------------------------------------------------------
## 5. Execution

#### 1. Data extraction from MIMIC-III and preporcessing
- Option 1 - Extracting data from Google Cloud using [Bigquery](https://console.cloud.google.com/bigquery)
	1. There are 5 sql files in code/big_query folder: pivoted_bg_art.sql, pivoted_gcs.sql, pivoted_lab.sql, vital_icustay_detail.sql and sepis_onset_time.sql.
	2. Run these queries on Google Bigquery console and download the results in csv format.
	3. Keep downloaded results in folder src/features/ (ie folder which has etl_spark.py)

- Option 2 - Use preporcessed data available [here](https://drive.google.com/drive/folders/1xbWL-nkx0a3FlDDZSZRemZQ6DVwzO8ed?usp=sharing)

	Folder has two csv files.  
		1. [Not-so-balanced dataset](https://drive.google.com/file/d/11xNns0a--cKfLROO73wOf0cG9Vo9MrgB/view?usp=sharing):  Case to control ratio was kept at 1:3.  
		2. [Highly imbalanced dataset](https://drive.google.com/file/d/1rh0cBrz1kekpWlRctQru54hDitFQWHZE/view?usp=sharing) : this dataset included all control patients. case to control ratio = 1:60


#### 2. Running the models
1. RNN : The train the lstm follow the steps below. You need to have [Anaconda](https://www.anaconda.com) already installed. 
```
>cd [PROJECT_ROOT]
>conda env create -f environment.yml
(sepsis_rediction) >cd scripts
(sepsis_rediction) >./train_lstm.sh
```

####Notes
* `src/features/etl_sepsis_data.py.` By default this extracts features from records in `data/data20191125.csv`. This will 
write train, validation and test data to `output/processed/{ids, labels, seqs}.{train, validation, test}`
* `src/rnn/train_variable_rnn.py` This creates datasets using data in `output/processed` and writes models to
     `output/lstm_enriched.path`.
* Model evaluation graphs and metrics from the training will be written to `doc/figs` as well as the logs in `stdout`.
* `train_variable_rnn.py:main` can be modified by commenting out the training step to only score the `*.path` file.
* `src/util/config.py` contains a lot of parameters which are used by LSTM model training.


2. Other Models : Run src/other_models/ac_models.py. This will run models using 'not-so-balanced' dataset and output following performance metrices for each model.
	  - Models - SVM , Logistic-Regression, Multi-Layer-Perceptron and Gradient-Boosting
	  - Metrics - Accuracy, F1-Score , Precision, Recall , AUC and Confusion-Matrix
	  
To run models using 'highly-imbalanced' dataset, select appropriate CASE_CTRL_FILE in src/other_models/ac_util.py.

```python
# Select appropriate dataset
CASE_CTRL_FILE   = DATA_DIR + "data20191125.csv"       # randomly sampled control patients (1:3)
#CASE_CTRL_FILE  = DATA_DIR + "data20191204.csv"       # Entire cohort  (1:60)
```
Sample output

![alt text](https://github.gatech.edu/tweldon7/sepsis_prediction/blob/master/doc/sample_output.png)


