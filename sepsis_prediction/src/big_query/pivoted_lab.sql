select subject_id, DATETIME_TRUNC(charttime,HOUR), avg(ANIONGAP) ANIONGAP ,avg(ALBUMIN) 	ALBUMIN,
avg(BANDS)	BANDS,
avg(BICARBONATE) BICARBONATE,	
avg(BILIRUBIN	) BILIRUBIN,
avg(CREATININE) CREATININE,	
avg(CHLORIDE) CHLORIDE	,
avg(GLUCOSE) GLUCOSE	,
avg(HEMATOCRIT) HEMATOCRIT,	
avg(HEMOGLOBIN) HEMOGLOBIN,	
avg(LACTATE) LACTATE	,
avg(PLATELET) PLATELET	,
avg(POTASSIUM) POTASSIUM	,
avg(PTT) PTT	,
avg(INR) INR	,
avg(PT) PT	,
avg(SODIUM) SODIUM	,
avg(BUN) BUN	,
avg(WBC) WBC

from `physionet-data.mimiciii_derived.pivoted_lab` as pv
group by subject_id, DATETIME_TRUNC(charttime,HOUR)