/* *****************************************************
 achauhan39 : Oct-21
 
 BigQuery to compute Sepsis onset time using criteria `Sofa >=2`
 
 Query makes use of following derived tables from schema 
 `physionet-data.mimiciii_derived`
	1. icustay_detail
	2. suspicion_of_infection
	3. pivoted_sofa

Copy the query in BigQuery editor and hit 'Run'.
https://console.cloud.google.com/bigquery
	
***************************************************** */

-- Get Infection Time
with inf as (
select * from (
	select * , ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY suspected_infection_time) as rn
	from `physionet-data.mimiciii_derived.suspicion_of_infection`   
	where suspected_infection_time is not NULL ) a
where rn = 1
),

-- Compute Sofa Window
sf as
(
    select ps.icustay_id, ps.starttime,ps.endtime , ps.SOFA_24hours,
    suspected_infection_time, intime,outtime,  W_start, W_end
    from `physionet-data.mimiciii_derived.pivoted_sofa` ps , 
    (
      SELECT inf.* , ie.intime, ie.outtime, admission_age,
          DATETIME_SUB(suspected_infection_time, INTERVAL 2 DAY) as T0,
          DATETIME_ADD(suspected_infection_time, INTERVAL 1 DAY) as T1,
          GREATEST(ie.intime , DATETIME_SUB(suspected_infection_time, INTERVAL 2 DAY)) as W_start,
          LEAST(ie.outtime, DATETIME_ADD(suspected_infection_time, INTERVAL 1 DAY) ) as W_end,
          DATETIME_DIFF(ie.outtime,ie.intime , HOUR) as icustay_hrs
      FROM
        -- `physionet-data.mimiciii_derived.suspinfect_poe` inf,  -- infection time
		  inf,  -- infection time
         `physionet-data.mimiciii_derived.icustay_detail` ie     -- icu event
      where 
        ie.icustay_id = inf.icustay_id 
	    --and ie.intime_hr is not NULL
        and inf.suspected_infection_time is not NULL
	    and DATETIME_DIFF(ie.outtime, ie.intime , HOUR) > 10  
		and DATETIME_DIFF(suspected_infection_time, ie.intime , HOUR) > 18  
		and admission_age > 15  
    ) sw  -- sofa_window`
    where 
    sw.icustay_id = ps.icustay_id
    and ps.starttime > sw.W_start
    and ps.endtime < sw.W_end
),

-- Get Sofa value at start of the window
starting_sofa as (
 select * from (
    select icustay_id, starttime,endtime ,SOFA_24hours as sofa_t0,
    ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime ) as rn
    from sf
  ) 
  where rn = 1
), 

sofa_delta as (

select sf.icustay_id, sf.starttime,sf.endtime, sf.SOFA_24hours, sofa_t0,
  (sf.SOFA_24hours - sofa_t0)as sofa_delta
from sf , starting_sofa
where 
  sf.icustay_id = starting_sofa.icustay_id
  and (sf.SOFA_24hours - sofa_t0 ) >= 2
),

onset_time as (
  select *,
  ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime ) as rn,
  starttime as sepsis_onset_time
  from sofa_delta
)

select * from onset_time
where rn = 1