# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:13:35 2019
@author: achauhan39
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from datetime import timedelta 
from sklearn.metrics import precision_score, recall_score, \
            accuracy_score, f1_score, roc_auc_score, roc_curve, auc,\
            confusion_matrix, classification_report            
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize, MaxAbsScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier

DATA_DIR   = "../../data/"
OUTPUT_DIR = "../../output/"    

# Select appropriate dataset
CASE_CTRL_FILE   = DATA_DIR + "data20191125.csv"       # randomly sampled control patients (1:3)
#CASE_CTRL_FILE  = DATA_DIR + "data20191204.csv"       # Entire cohort  (1:60)


SEPSIS_SUPERSET_FILE = "../data/sepsis_superset_patientIDs.csv"

GCS = ['GCS','GCSMotor','GCSVerbal','GCSEyes','EndoTrachFlag']
BG  = ['SPECIMEN_PROB','SO2','spo2','PO2','PCO2','fio2_chartevents','FIO2','AADO2','AADO2_calc','PaO2FiO2Ratio','PH','BASEEXCESS','TOTALCO2', \
       'CARBOXYHEMOGLOBIN','METHEMOGLOBIN','CALCIUM','TEMPERATURE','INTUBATED','TIDALVOLUME','VENTILATIONRATE','VENTILATOR','PEEP','O2Flow','REQUIREDO2'  ]
LAB = ['ANIONGAP','ALBUMIN','BANDS','BICARBONATE','BILIRUBIN','CREATININE','CHLORIDE','GLUCOSE','HEMATOCRIT','HEMOGLOBIN','LACTATE','PLATELET','POTASSIUM','PTT','INR','PT','SODIUM','BUN','WBC' ]

NEW_FEATURES = GCS + LAB 



TEST_SIZE       = 0.20
VAL_SIZE        = 0.10
RANDOM_STATE    = 100

np.random.seed(RANDOM_STATE)


    
def pca_3d(X,Y) :
    pca = PCA(n_components= 10, random_state=10)   #n_components
    X = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    df = pd.DataFrame(np.hstack((X, np.atleast_2d(Y).T)))
    df = (df - df.mean())/(df.max() - df.min())
    
    cols = list(range(df.shape[1]))
    cols[-1] = 'label'
    df.columns = cols
    #df.to_csv('pca_dataset.csv', index=False)
    #df = pd.read_csv('pca_dataset.csv', )  
    
    fig = plt.figure()
    title_obj = plt.title("PCA")
    plt.setp(title_obj, color='b')
    ax = Axes3D(fig)
    ax.scatter(df[0], df[1], df[2] , c=df['label'], marker='o' , alpha=1)
    ax.set_xlabel('PC-1')
    ax.set_ylabel('PC-2')
    ax.set_zlabel('PC-3')
    plt.show()

   # %matplotlib qt
    
   
   
def feature_imp(trainX, trainY , df):
    
    feat_names = df.columns #[:-1]
    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(trainX,trainY)
    importances = forest.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    for f in range(trainX.shape[1]):
        print("%2d) %-*s %f" % (f+1 , 30,
            feat_names[indices[f]],
            importances[indices[f]]))

    plt.title('Feature Importances')
    plt.bar(range(trainX.shape[1]),
        importances[indices],
        color='blue',
        align='center')
    plt.xticks(range(trainX.shape[1]),
               feat_names[indices], rotation=90)
    plt.xlim([-1, trainX.shape[1]])
    plt.tight_layout()

    # select top features
    model = SelectFromModel(forest, prefit=True)
    X_new = model.transform(trainX)
    print(X_new[:5,]) 



def tune_it(clf, X,Y, params ,clfname=""):
    
    X_train, X_val, Y_train,Y_val = ms.train_test_split(X,Y, test_size = VAL_SIZE , stratify = Y,random_state = RANDOM_STATE)    
    
    gs = ms.GridSearchCV(clf, param_grid =params, scoring ='roc_auc',cv=5, return_train_score=True)            
    gs.fit(X_train, Y_train)
    
    #res_df = pd.DataFrame.from_dict(gs.cv_results_)
    #res_df.to_csv('output/'+ clfname +'.csv')
    print(gs.best_params_)
    
    best_clf = gs.best_estimator_
    best_clf.fit(X_train, Y_train)
    
#    print (clfname)
#    print('Train Auc: %.4f' % best_clf.score(X_train, Y_train))
#    print('Validation Auc: %.4f' % best_clf.score(X_val, Y_val))
    
    return best_clf 



def plot_roc(fpr,tpr,clf_name):
       # Calculate the AUC
        roc_auc = auc(fpr, tpr)
    
        # Plot of a ROC curve for a specific class
        #plt.figure()
        title_obj = plt.title(clf_name + ' : ROC ')
        plt.setp(title_obj, color='b')
        plt.plot(fpr, tpr, label='(auc = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title()
        plt.legend(loc="lower right")
        plt.show()
        

def plot_confusion_matrix(y_true, y_pred, clf_name=""):

    class_names =['no-sepsis','sepsis']         
    cm = confusion_matrix(y_true, y_pred)
#    print(cm)
    # Normalize CM values
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
#    print(cm)
    

    fig, ax = plt.subplots(figsize=(2,2))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #ax.figure.colorbar(im, ax=ax)
 
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title = clf_name ,
           ylabel='True',
           xlabel='Predicted')

    ax.xaxis.label.set_color('blue')
    ax.yaxis.label.set_color('blue')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center" , rotation_mode="anchor", color='b')
    plt.setp(ax.get_xticklabels(), color='b')
    
    plt.show()

    
def get_scores(clf, X ,Y , dataset="" , clf_name="" ):
    y_pred   = clf.predict(X)
    return calc_scores(Y, y_pred, dataset, clf_name)

def calc_scores(Y, y_pred, dataset="", clf_name="", digit=6):
  
    acc = round(accuracy_score(Y, y_pred ), digit)
#    f1       = round(f1_score(Y, y_pred, average='weighted'), digit)
#    recall   = round(recall_score(Y, y_pred, average='weighted'), digit)
#    prec     = round(precision_score(Y, y_pred, average='weighted'), digit)
  
    f1       = round(f1_score(Y, y_pred), digit)
    recall   = round(recall_score(Y, y_pred), digit)
    prec     = round(precision_score(Y, y_pred), digit)
    
    auc_val  = round(roc_auc_score(Y, y_pred),digit)

    print ('\t ', dataset,' - ',  clf_name.upper())   
    print ('Accuracy :', acc)                                 
    print ('F1 score :', f1)
    print ('Recall   :', recall)
    print ('Precision:', prec)
    print ('AUC      :', auc_val)
    
    fpr, tpr, thresholds = roc_curve(Y, y_pred)
#    plot_roc(fpr,tpr,clf_name)
    plot_confusion_matrix (Y, y_pred , clf_name)   
    
    print ("\n -------------------------------------------------------------------\n")


def eval_model(clf, X_train, X_test, Y_train , Y_test , clfname="" ):
    get_scores(clf, X_train, Y_train, "Train" , clfname)
    get_scores(clf, X_test,  Y_test, "Test" , clfname)
    
    
def filter_data_ac(pred_w, obsv_w) :
    
    CASE_FILE = "../data/case_patient_final.csv"
    CTRL_FILE = "../data/control_patient_final.csv"

    # 1. Read case & control files. Index_date = last_event/sepsis_onset_time - prediction_window
    print ("reading in  ", CASE_FILE)
    
    df_case = pd.read_csv( CASE_FILE , parse_dates = ['sepsis_onset_time','date_hour'])
    df_case['index_date'] = df_case['sepsis_onset_time'] - timedelta(hours= pred_w)
    
    df_ctrl = pd.read_csv( CTRL_FILE , parse_dates = ['last_event','date_hour'])
    df_ctrl['index_date'] = df_ctrl['last_event']
    
    df = pd.concat([df_case, df_ctrl])
    df['label'] = np.where(df['type']=='CASE', 1, 0)
    df['gender'] = np.where(df['gender']=='M', 1, 0)
    
    # 2. filter data for observation window. Currently set to 12 hrs
    cols = ['icustay_id', 'date_hour', 'avg_HeartRate', 'avg_SysBP', 'avg_DiasBP',
           'avg_MeanBP', 'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose',
           'admission_age', 'gender','index_date' , 'label']  #'type'. 'first_icu_stay'
    
    df = df[cols]
    df['diff'] = df['index_date'] - df['date_hour']
    df = df.loc[ (df['diff'] >= timedelta(hours=0)) & (df['diff'] <= timedelta(hours= obsv_w)) ]
    
    # compute number of observations in Obsv_window
    df['cnt'] = 1
    df['cnt'] = df.groupby('icustay_id')['cnt'].transform( lambda x : x.sum()) 
    
    # Filter out icustay_id if number of hourly observations < 3
    df = df[df['cnt'] >= 4]
    
    check_patient_cnt(df , 'label') 
    
    return df
    


def check_patient_cnt(df , col = 'CaseControl' ) :   
    #df.groupby(['icustay_id']).count()      # 2716
    if col == 'CaseControl':
        case = df[df['CaseControl'] == 'Case']
        ctrl = df[df['CaseControl'] == 'Control']   
    else :
        case = df[df['label'] == 1]
        ctrl = df[df['label'] == 0]    
        
    print( " Final : case# = {} , ctrl# = {} " .format(case['icustay_id'].nunique() , ctrl['icustay_id'].nunique()))
    
    #pd.crosstab(df.icustay_id, df.label)  
    #df['Pred or Obs'].value_counts()        # Prediction : 424149 | Observation : 17137
    #df.groupby(['CaseControl']).count()     # only case patients are showing up

    
def check_exlcusion(df_ctrl) :
    exclude = pd.read_csv(SEPSIS_SUPERSET_FILE)
    
    A = set(exclude.subject_id)
    B = set(df_ctrl.subject_id) 
    C = set.intersection(A,B)           
    
    if len(C)> 0 :
        print("Error : {} Sepsis patients found in Control group. ".format(len(C)) )
        df_ctrl = pd.merge(df_ctrl, exclude, on='subject_id' , how ='left')
        df_ctrl['count'].fillna(0, inplace= True)
        df_ctrl = df_ctrl[ df_ctrl['count'] == 0]
        B = set(df_ctrl.subject_id) 
        C = set.intersection(A,B)			# should be zero
    return df_ctrl


def preprocess(pred_w, obsv_w) :
    
    print ("reading in " , CASE_CTRL_FILE )    
    df = pd.read_csv(CASE_CTRL_FILE, 
                     parse_dates = (['f0_','sepsis_onset_time' , 'intime' , 'outtime' ]))
            
    cols = [ 'subject_id', 'icustay_id', 'f0_' ,'avg_HeartRate', 'avg_SysBP', 'avg_DiasBP',
       'avg_MeanBP', 'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose', 'admission_age', 'gender' ,     
        'sepsis_onset_time' , 'intime' , 'outtime' , 'los_icu','CaseControl']
 
    cols = cols + NEW_FEATURES
    
    df = df[cols]
    df.rename( columns= { 'f0_' : 'date_hour'}, inplace= True)
    
    df_case = df[df['CaseControl'] == 'Case']
    df_ctrl = df[df['CaseControl'] == 'Control']   
    print( "Initial : case# = {} , ctrl# = {} " .format(df_case['icustay_id'].nunique() , df_ctrl['icustay_id'].nunique()))
              
    # remove control patients if flagged by other Sepsis criteria (Angus, Martin, CDC , ICD-code)
    #df_ctrl = check_exlcusion(df_ctrl)

    
    # select case patients where sepsis_onset_time > intime
    df_case = df_case.loc[ (df_case['sepsis_onset_time']- df['intime']) >= timedelta(hours=0)]

    
    df_case['index_date'] = df_case['sepsis_onset_time'] - timedelta(hours= pred_w)
    
    # get last event date for control patients
    last_event = lambda x : x['date_hour'].max()
    tmp  = df_ctrl.groupby('icustay_id').apply(last_event).reset_index()
    tmp.rename(columns={0 : 'index_date'}, inplace=True)
    df_ctrl = pd.merge(df_ctrl, tmp, on='icustay_id' , how='left' )

    df = pd.concat([df_case, df_ctrl])
    df['label'] = np.where(df['CaseControl']=='Case', 1, 0)
    df['gender'] = np.where(df['gender']=='M', 1, 0)
    
    # select data for observation window
    df['diff'] = df['index_date'] - df['date_hour']
    df = df.loc[ (df['diff'] >= timedelta(hours=0)) & (df['diff'] <= timedelta(hours= obsv_w)) ]
    
    # compute number of observations in Obsv_window
    df['cnt'] = 1
    df['cnt'] = df.groupby('icustay_id')['cnt'].transform( lambda x : x.sum()) 
    
    # Filter out icustay_id if number of hourly observations <= 3
    df = df[df['cnt'] >= 3]
    
    check_patient_cnt(df)
          
    # remove unnecessary columns  
    df = df.drop( [ 'index_date' , 'diff' , 'sepsis_onset_time', 'intime','outtime', 'los_icu', 'subject_id','CaseControl'] , axis=1 )
    
    return df


def create_features(df):
    df.isnull().sum()           
    cols = ['avg_HeartRate', 'avg_SysBP', 'avg_DiasBP', 'avg_MeanBP', 'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose']
    cols = cols + NEW_FEATURES
    
    df.loc[:, cols] = df.loc[:, cols].ffill()
    df.loc[:, cols] = df.loc[:, cols].bfill()
    df.isnull().sum()
#    df['shock_index'] = df['avg_HeartRate']/df['avg_SysBP'] 
#    df['bun_creatinine'] = df['BUN']/df['CREATININE']
#    
#    cols  = cols + ['shock_index' , 'bun_creatinine' ]
    
    my_dict ={}
    for i in cols :
        my_dict[i] = [np.mean , np.ptp]
    
    my_dict['admission_age'] = np.mean
    my_dict['label'] = np.max
        
    df = df.groupby(['icustay_id']).agg(my_dict)
    df.columns = [ "_".join(x) for x in df.columns]
    df.rename(columns ={'admission_age_mean':'admission_age', 'gender_max' : 'gender' , 'label_amax': 'label' } , inplace = True)
    
    # fix Age to 95th percentile. Some entries are 300+
    df.admission_age.describe()
    np.quantile(df.admission_age, .96)
    df.set_value(df['admission_age']>100, ['admission_age'], 100)
    
    # shuffle the data     
    df = df.sample(frac=1, random_state = 10)
    df.label.value_counts()
    
    return df
    

def get_sepsis_train_test_data(pred_w = 6 , obsv_w = 12 ) :
    
    # df = filter_data_ac(pred_w, obsv_w)    
    df = preprocess(pred_w, obsv_w ) 
    
    df = create_features(df)
    df = df.fillna(0)
    
    # create train/test set
    df = df.apply(pd.to_numeric)
    Y = df.label.values
    del df['label']
    X = df.values.astype(np.float64)
    
    X_train, X_test, Y_train , Y_test = ms.train_test_split(X, Y, test_size= TEST_SIZE, stratify=Y , random_state = RANDOM_STATE )
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)
    
    print('TrainX :', X_train.shape , " |  trainY : ", Y_train.shape )
    print('TestX :', X_test.shape ,   " |  TestY : ", Y_test.shape )
    
    #np.unique(Y_test, return_counts = True)
    #np.unique(Y_train, return_counts = True)
    
    # feature_imp(X_train, Y_train , df)
#    sel = SelectFromModel(RandomForestClassifier( n_estimators = 100))
#    sel.fit(X_train, Y_train)
#    selected_feat= df.columns[(sel.get_support())]
#    print ("Features Selected -----------" )
#    print(selected_feat)
    
    
    return X_train, X_test, Y_train , Y_test



def process_note_data():
    
    df_note = pd.read_csv("../data/text_features_1575446631.csv" , parse_dates = ['CHARTTIME'])
    df_note = df_note.dropna(subset =['CHARTTIME'] )
    cols = [c.lower() for c in df_note.columns ]
    df_note.columns = cols 
    
  
    df_id = pd.read_csv("../data/cohort_list.csv" , parse_dates= ['date_hour'])
    df_id['dt_min'] = df_id.groupby('subject_id')['date_hour'].transform(lambda x : x.min())
    df_id['dt_max'] = df_id.groupby('subject_id')['date_hour'].transform(lambda x : x.max())
    
    
    cols = ['subject_id', 'dt_min' , 'dt_max' , 'label']
    cohort = df_id[cols].drop_duplicates()
    
    df_merge = pd.merge(df_note, cohort, on ='subject_id' , how= 'inner' )
    df_merge = df_merge[ (df_merge.charttime >= df_merge.dt_min) &  (df_merge.charttime <= df_merge.dt_max) ]
    
    df_merge = df_merge[ ['subject_id', 'text', 'label'] ]

    df_t = df_merge.groupby('subject_id').agg({'text' : ' '.join  , 'label' : 'max' })

















    # Roll up all date_hours into 1 row per patient - mean,median , min , max ??
#    df = df.groupby(['icustay_id']).agg({ 
#           'avg_HeartRate'  : [np.mean , np.ptp], 
#           'avg_SysBP'      : [np.mean , np.ptp], 
#           'avg_DiasBP'     : [np.mean , np.ptp], 
#           'avg_MeanBP'     : [np.mean , np.ptp], 
#           'avg_RespRate'   : [np.mean , np.ptp], 
#           'avg_TempC'      : [np.mean , np.ptp], 
#           'avg_SpO2'       : [np.mean , np.ptp],  
#           'avg_Glucose'    : [np.mean , np.ptp],   
#           'shock_index'    : [np.mean , np.ptp],
#           'GCS'            : [np.mean , np.ptp],
#
#           'bun_creatinine' : [np.mean , np.ptp],
##           'WBC'            : [np.mean , np.ptp],
##           'BILIRUBIN'      : [np.mean , np.ptp],
#           
##           'PaO2FiO2Ratio'  : [np.mean , np.ptp],
##           'PCO2'           : [np.mean , np.ptp],
#           
##           'REQUIREDO2'     : [np.mean , np.ptp],
##           'METHEMOGLOBIN'  : [np.mean , np.ptp],
#                    
#           'admission_age' : np.mean,
#           'label' : np.max  })


