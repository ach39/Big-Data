select icustay_id, DATETIME_TRUNC(charttime,HOUR) charttime, avg(GCS) GCS	,
avg(GCSMotor) GCSMotor,
avg(GCSVerbal) GCSVerbal	,
avg(GCSEyes) GCSEyes,	
avg(EndoTrachFlag) EndoTrachFlag

from `physionet-data.mimiciii_derived.pivoted_gcs` as pv
group by icustay_id, DATETIME_TRUNC(charttime,HOUR)