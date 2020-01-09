select * from (select subject_id, hadm_id, gender, dod, admittime, dischtime, los_hospital, 
admission_age, ethnicity, admission_type, hospital_expire_flag, hospstay_seq, first_hosp_stay, intime, outtime, los_icu,
icustay_seq, first_icu_stay, g.*
from 
(
select 
icustay_id, DATETIME_TRUNC(charttime, HOUR),
avg(HeartRate) avg_HeartRate, 
avg(SysBP) avg_SysBP, 
avg(DiasBP) avg_DiasBP, 
avg(MeanBP) avg_MeanBP, 
avg(RespRate) avg_RespRate, 
avg(TempC) avg_TempC, 
avg(SpO2) avg_SpO2, 
avg(Glucose) avg_Glucose
 
from `physionet-data.mimiciii_derived.pivoted_vital` as pv 

group by icustay_id, DATETIME_TRUNC(charttime, HOUR) 
order by 1,2
) as g
join `physionet-data.mimiciii_derived.icustay_detail` as id 
using (icustay_id)
order by 1,2
) as A
Where A.admission_age >= 15;