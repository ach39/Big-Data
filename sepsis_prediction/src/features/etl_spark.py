# importing libraries
from pyspark.sql import HiveContext, SparkSession
from pyspark.sql.functions import count
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
sparkSession = SparkSession.builder.appName("example-pyspark-read-and-write").getOrCreate()

# df is the full dataset of all the patients
df = sparkSession.read.csv('G:\\Sepsis_data.csv', header="true")

# list of all the case Other Criteria and Sofa Criteria Case patients
list_all_case_patients = sparkSession.read.csv('G:\\sepsis_superset_patientIDs.csv', header="true")
list_all_case_subject_ids = list_all_case_patients.select("subject_id").withColumn('CaseControl', F.lit('Case'))

# list all the distinct subject_id
list_all_subject_ids = df.select("subject_id").distinct()

# Defining Control and Case patients by simple left join on subject ID
df_control_case = list_all_subject_ids.join(list_all_case_subject_ids, 'subject_id', 'left').fillna({'CaseControl':'Control'})

#Control DF
############################################################################################
df_control_1 = df.join(df_control_case,"subject_id").filter(F.col("CaseControl") == "Control")
# Random sampling in Control DF
distinct_control = df_control_1.select("subject_id").distinct().orderBy(F.rand(1234)).limit(1360)
# Final_control
df_control = df_control_1.join(distinct_control,"subject_id").withColumn("sepsis_onset_time",F.col("outtime"))
############################################################################################

#Case DF
############################################################################################
df_case_1 = df.join(df_control_case,"subject_id").filter(F.col("CaseControl") == "Case")

# Manipulating the case
list_case_patients = sparkSession.read.csv('G:\\sepsis_onset_time.csv', header="true")

#Final Case
df_case = df_case_1.join(list_case_patients.select('icustay_id', 'sepsis_onset_time'), 'icustay_id')
############################################################################################

############################################################################################
# Taking Union of the data
df_union_case_control_icustay_id = df_control.select("icustay_id").union(df_case.select("icustay_id")).distinct()

DF = df.join(df_union_case_control_icustay_id, "icustay_id").join(df_control_case, "subject_id")
############################################################################################


# Adding in Index Date and seperating test and train
############################################################################################
# Index Date Derivation
DF_sepsis_date = DF.join(list_case_patients.select('icustay_id', 'sepsis_onset_time').distinct(), 'icustay_id', "left")
DF_index_date = DF_sepsis_date.withColumn("Index_Date",F.when(F.col("sepsis_onset_time").isNull(), F.col("outtime")).otherwise(F.col("sepsis_onset_time")))

# Train and Test Data
train, test = DF_index_date.select("icustay_id").distinct().randomSplit([0.7, 0.3], seed=12345)
train_df = train.withColumn("TestTrain", lit("Train"))
test_df = test.withColumn("TestTrain", lit("Test"))
DF_TRAIN_TEST = DF_index_date.join(train_df.union(test_df),'icustay_id')

# # Adding the prediction window and observation window
# DF_PRED_OBV = DF_TRAIN_TEST.withColumn("index_hrs",((F.hour("Index_Date")-F.hour("f0_"))+ F.datediff(F.col("f0_"),F.col("Index_Date"))*24))
# seperation_time = "-18" # Change the hours if we want to or we can discard the data as well
# DF_PRED_OBV.filter(F.col("Index_Date") >= F.col("f0_"))
# DF_PRED_OBV = DF_PRED_OBV.filter((F.col("index_hrs") >= "0")& (F.col("index_hrs") <= seperation_time))
# DF_PRED_OBV = DF_PRED_OBV.withColumn("Pred or Obs", F.when((F.col("index_hrs") > "6") , "Observation").otherwise("Prediction"))



############################################################################################


# Fill in the null values with the Average value of each score
stat = "mean"
DF_mean = DF_TRAIN_TEST.groupBy("icustay_id").agg({"avg_HeartRate":stat,"avg_SysBP":stat,"avg_DiasBP":stat,"avg_MeanBP":stat,\
                                                       "avg_RespRate":stat,"avg_TempC":stat,"avg_SpO2":stat,"avg_Glucose":stat,})

Final_df_nullfix = DF_TRAIN_TEST.join(DF_mean, ["icustay_id"])

Final_df_nullfix = Final_df_nullfix.withColumn("AVG HEARTRATE",F.when(F.col("avg_HeartRate").isNull(),F.col("avg(avg_HeartRate)")).otherwise( F.col("avg_HeartRate")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG SYSBP",F.when(F.col("avg_SysBP").isNull(),F.col("avg(avg_SysBP)")).otherwise( F.col("avg_SysBP")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG DIASBP",F.when(F.col("avg_DiasBP").isNull(),F.col("avg(avg_DiasBP)")).otherwise( F.col("avg_DiasBP")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG MEANBP",F.when(F.col("avg_MeanBP").isNull(),F.col("avg(avg_MeanBP)")).otherwise( F.col("avg_MeanBP")))

Final_df_nullfix = Final_df_nullfix.withColumn("AVG RESPRATE",F.when(F.col("avg_RespRate").isNull(),F.col("avg(avg_RespRate)")).otherwise( F.col("avg_RespRate")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG TEMPC",F.when(F.col("avg_TempC").isNull(),F.col("avg(avg_TempC)")).otherwise( F.col("avg_TempC")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG SPO2",F.when(F.col("avg_SpO2").isNull(),F.col("avg(avg_SpO2)")).otherwise( F.col("avg_SpO2")))
Final_df_nullfix = Final_df_nullfix.withColumn("AVG GLUCOSE",F.when(F.col("avg_Glucose").isNull(),F.col("avg(avg_Glucose)")).otherwise( F.col("avg_Glucose")))



# Selecting necessary columns
Final_df_nullfix = Final_df_nullfix.select(['icustay_id', 'subject_id', 'hadm_id', 'gender', 'dod', 'admittime', 'dischtime', 'los_hospital', 'admission_age', 'ethnicity', 'admission_type', 'hospital_expire_flag', 'hospstay_seq', 'first_hosp_stay', 'intime', 'outtime', 'los_icu', 'icustay_seq', 'first_icu_stay', 'f0_', 'avg_HeartRate', 'avg_SysBP', 'avg_DiasBP', 'avg_MeanBP', 'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose', 'sepsis_onset_time', 'CaseControl', 'TestTrain', 'Index_date',  'AVG HEARTRATE', 'AVG SYSBP', 'AVG DIASBP', 'AVG MEANBP', 'AVG RESPRATE', 'AVG TEMPC', 'AVG SPO2', 'AVG GLUCOSE']
)



#######################################################################################
# Cleaning Null Values with average in the column
def fill_with_mean(df, exclude=set()):
    stats = df.agg(*(
        F.mean(c).alias(c) for c in df.columns if c not in exclude
    ))
    return df.na.fill(stats.first().asDict())

Final_df_nullfix = fill_with_mean(Final_df_nullfix, ['icustay_id', 'subject_id', 'hadm_id', 'gender', 'dod', 'admittime', 'dischtime', 'los_hospital', 'admission_age', 'ethnicity', 'admission_type', 'hospital_expire_flag', 'hospstay_seq', 'first_hosp_stay', 'intime', 'outtime', 'los_icu', 'icustay_seq', 'first_icu_stay', 'f0_', 'avg_HeartRate', 'avg_SysBP', 'avg_DiasBP', 'avg_MeanBP', 'avg_RespRate', 'avg_TempC', 'avg_SpO2', 'avg_Glucose', 'sepsis_onset_time', 'CaseControl', 'TestTrain', 'Index_date'])




#######################################################################################
# Adding additional dataset
pivoted_gcs = sparkSession.read.csv('G:\\pivoted_gcs.csv', header="true")
pivoted_lab = sparkSession.read.csv('G:\\pivoted_lab.csv', header="true")
pivoted_bg_art = sparkSession.read.csv('G:\\pivoted_bg_art.csv', header="true")
#######################################################################################

# FINAL_DF_BG = Final_df_nullfix.join(pivoted_bg, (Final_df_nullfix.icustay_id == pivoted_bg.icustay_id) & (Final_df_nullfix.f0_ == pivoted_bg.charttime), how = 'left').drop(pivoted_bg.icustay_id)
# FINAL_DF_BG = FINAL_DF_BG.drop("charttime")
#pivoted_bg_art = pivoted_bg_art.drop("hadm_id")

FINAL_DF_GCS = Final_df_nullfix.join(pivoted_gcs, (Final_df_nullfix.icustay_id == pivoted_gcs.icustay_id) & (Final_df_nullfix.f0_ == pivoted_gcs.charttime), how = 'left').drop(pivoted_gcs.icustay_id).drop(pivoted_gcs.charttime)


FINAL_DF_BG_ART = FINAL_DF_GCS.join(pivoted_bg_art, (FINAL_DF_GCS.icustay_id == pivoted_bg_art.icustay_id) & (FINAL_DF_GCS.f0_ == pivoted_bg_art.charttime), how='left').drop(pivoted_bg_art.icustay_id).drop(pivoted_bg_art.charttime).drop(pivoted_bg_art.LACTATE).drop(pivoted_bg_art.HEMOGLOBIN).drop(pivoted_bg_art.BICARBONATE).drop(pivoted_bg_art.SODIUM).drop(pivoted_bg_art.HEMATOCRIT).drop(pivoted_bg_art.POTASSIUM).drop(pivoted_bg_art.GLUCOSE).drop(pivoted_bg_art.CHLORIDE)

FINAL_DF_LAB = FINAL_DF_BG_ART.join(pivoted_lab, (FINAL_DF_BG_ART.subject_id == pivoted_lab.subject_id) & (FINAL_DF_BG_ART.f0_ == pivoted_lab.f0_), how = 'left').drop(pivoted_lab.f0_).drop(pivoted_lab.subject_id)


#FINAL_DF_BG_ART = fill_with_mean(FINAL_DF_BG_ART, Final_df_nullfix.columns)

# Adding the prediction and observation window
FINAL_DF_LAB = FINAL_DF_LAB.withColumn("index_hrs",((F.hour("Index_Date")-F.hour("f0_"))+ F.datediff(F.col("Index_Date"),F.col("f0_"),)*24))
seperation_time = "18"
FINAL_DF_LAB = FINAL_DF_LAB.filter((F.col("index_hrs") >= "0")& (F.col("index_hrs") <= seperation_time))
FINAL_DF_LAB = FINAL_DF_LAB.withColumn("Pred or Obs", F.when((F.col("index_hrs") > "6") , "Observation").otherwise("Prediction"))

# Writing CSV

FINAL_DF_LAB.repartition(1).write.format('com.databricks.spark.csv').save("ETL_sepsis_data_20191125.csv",header = 'true')
