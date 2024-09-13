create database coal_forecast;
use coal_forecast;
/* create table coal and import data from the csv file by using data import wizard .*/

/*show the records in the coal table*/
select * from coal;

/* Describe the table to see it's structure.*/
describe coal;

/* perform first moment business decission ( mean, median,mode)*/
select AVG(Coal_RB_4800_FOB_London_Close_USD) as mean_value1 from coal;
select AVG(Coal_RB_5500_FOB_London_Close_USD) as mean_value2 from coal;
select avg(Coal_RB_5700_FOB_London_Close_USD) as mean_value3 from coal;
select AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) as mean_value4 from coal;
select AVG(Coal_India_5500_CFR_London_Close_USD) as mean_value5 from coal;
select AVG(Price_WTI) as mean_value6 from coal;
select AVG(Price_Brent_Oil) as mean_value7 from coal;
select avg(Price_Dubai_Brent_Oil) as mean_value8 from coal;
select AVG(Price_ExxonMobil) as mean_value9 from coal;
select AVG(Price_Shenhua) as mean_value10 from coal;
select AVG(Price_All_Share) as mean_value11 from coal;
select AVG(Price_Mining) as mean_value12 from coal;
select AVG(Price_LNG_Japan_Korea_Marker_PLATTS) as mean_value13 from coal;
select avg(Price_ZAR_USD) as mean_value14 from coal;
select AVG(Price_Natural_Gas) as mean_value15 from coal;
select AVG(Price_ICE) as mean_value16 from coal;
select avg(Price_Dutch_TTF) as mean_value17 from coal;
select avg(Price_Indian_en_exg_rate) as mean_value18 from coal;

/* -------------------------------------------median-----------------------------------*/

/*-------------------------------------------------Coal_RB_4800_FOB_London_Close_USD----------------------*/
/* add row index to the 1st column*/
SET @rowindex := -1;

SELECT AVG(N.Coal_RB_4800_FOB_London_Close_USD) as median1
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
           coal.Coal_RB_4800_FOB_London_Close_USD 
    FROM coal
    ORDER BY coal.Coal_RB_4800_FOB_London_Close_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

/*--------------------------------------Coal_RB_5500_FOB_London_Close_USD--------------------------------------------*/

SET @rowindex := -1;
SELECT AVG(N.Coal_RB_5500_FOB_London_Close_USD) as median2
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
           Coal_RB_5500_FOB_London_Close_USD
    FROM coal
    ORDER BY Coal_RB_5500_FOB_London_Close_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

/*-------------------------Coal_RB_5700_FOB_London_Close_USD----------------*/


SET @rowindex := -1;
SELECT AVG(N.Coal_RB_5700_FOB_London_Close_USD) as median3
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
          Coal_RB_5700_FOB_London_Close_USD
    FROM coal
    ORDER BY Coal_RB_5700_FOB_London_Close_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

-------------------------------------------------Coal_RB_6000_FOB_CurrentWeek_Avg_USD--------------------------------------------------


SET @rowindex := -1;
SELECT AVG(N.Coal_RB_6000_FOB_CurrentWeek_Avg_USD) as median4
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Coal_RB_6000_FOB_CurrentWeek_Avg_USD
    FROM coal
    ORDER BY Coal_RB_6000_FOB_CurrentWeek_Avg_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

----------------------------------------------Coal_India_5500_CFR_London_Close_USD---------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Coal_India_5500_CFR_London_Close_USD) as median5
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Coal_India_5500_CFR_London_Close_USD
    FROM coal
    ORDER BY Coal_India_5500_CFR_London_Close_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);


-------------------------------------------------Price_WTI-------------------------------------------------------------


SET @rowindex := -1;
SELECT AVG(N.Price_WTI) as median6
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_WTI
    FROM coal
    ORDER BY Price_WTI
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

--------------------------------------------------Price_Brent_Oil------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Brent_Oil) as median7
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
        Price_Brent_Oil
    FROM coal
    ORDER BY Price_Brent_Oil
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

---------------------------------------Price_Dubai_Brent_Oil------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Dubai_Brent_Oil) as median8
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_Dubai_Brent_Oil
    FROM coal
    ORDER BY Price_Dubai_Brent_Oil
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

--------------------------------------------Price_ExxonMobil--------------------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_ExxonMobil) as median9 
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_ExxonMobil
    FROM coal
    ORDER BY Price_ExxonMobil
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

-----------------------------------------Price_Shenhua-------------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Shenhua) as median10
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_Shenhua
    FROM coal
    ORDER BY Price_Shenhua
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

--------------------------------------------Price_All_Share------------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_All_Share) as median11
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
        Price_All_Share
    FROM coal
    ORDER BY Price_All_Share
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

-----------------------------------------------Price_Mining--------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Mining) as median12
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_Mining
    FROM coal
    ORDER BY Price_Mining
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

----------------------------------------------------------Price_LNG_Japan_Korea_Marker_PLATTS----------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_LNG_Japan_Korea_Marker_PLATTS) as median13
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_LNG_Japan_Korea_Marker_PLATTS
    FROM coal
    ORDER BY Price_LNG_Japan_Korea_Marker_PLATTS
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);


-------------------------------------------Price_ZAR_USD-----------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_ZAR_USD) as median14
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_ZAR_USD
    FROM coal
    ORDER BY Price_ZAR_USD
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

-------------------------------------------Price_Natural_Gas---------------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Natural_Gas) as median15
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_Natural_Gas
    FROM coal
    ORDER BY Price_Natural_Gas
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

------------------------------------------------------Price_ICE----------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_ICE) as median16
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
        Price_ICE
    FROM coal
    ORDER BY Price_ICE
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

--------------------------------------------------------Price_Dutch_TTF-----------------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Dutch_TTF) as median17
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
        Price_Dutch_TTF
    FROM coal
    ORDER BY Price_Dutch_TTF
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);

------------------------------------------Price_Indian_en_exg_rate---------------------------------------

SET @rowindex := -1;
SELECT AVG(N.Price_Indian_en_exg_rate) as median18
FROM (
    SELECT @rowindex := @rowindex + 1 AS rowindex, 
         Price_Indian_en_exg_rate
    FROM coal
    ORDER BY Price_Indian_en_exg_rate
) AS N
WHERE N.rowindex = FLOOR(@rowindex / 2) OR N.rowindex = CEIL(@rowindex / 2);


/*-----------------------------------------------------MODE-------------------------------------------------------*/

SELECT Coal_RB_4800_FOB_London_Close_USD, COUNT(*) AS frequency
FROM coal
GROUP BY Coal_RB_4800_FOB_London_Close_USD
ORDER BY frequency DESC
LIMIT 1;


SELECT Coal_RB_5500_FOB_London_Close_USD ,COUNT(*) AS frequency
FROM coal
GROUP BY Coal_RB_5500_FOB_London_Close_USD
ORDER BY frequency DESC
LIMIT 1;


SELECT Coal_RB_5700_FOB_London_Close_USD, COUNT(*) AS frequency
FROM coal
GROUP BY Coal_RB_5700_FOB_London_Close_USD
ORDER BY frequency DESC
LIMIT 1;


SELECT Coal_RB_6000_FOB_CurrentWeek_Avg_USD, COUNT(*) AS frequency
FROM coal
GROUP BY Coal_RB_6000_FOB_CurrentWeek_Avg_USD
ORDER BY frequency DESC
LIMIT 1;


SELECT Coal_India_5500_CFR_London_Close_USD, COUNT(*) AS frequency
FROM coal
GROUP BY Coal_India_5500_CFR_London_Close_USD
ORDER BY frequency DESC
LIMIT 1;



SELECT Price_WTI, COUNT(*) AS frequency
FROM coal
GROUP BY Price_WTI
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_Brent_Oil, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Brent_Oil
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_Dubai_Brent_Oil ,COUNT(*) AS frequency
FROM coal
GROUP BY Price_Dubai_Brent_Oil
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_ExxonMobil, COUNT(*) AS frequency
FROM coal
GROUP BY Price_ExxonMobil
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_Shenhua, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Shenhua
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_All_Share, COUNT(*) AS frequency
FROM coal
GROUP BY Price_All_Share
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_Mining, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Mining
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_LNG_Japan_Korea_Marker_PLATTS, COUNT(*) AS frequency
FROM coal
GROUP BY Price_LNG_Japan_Korea_Marker_PLATTS
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_ZAR_USD, COUNT(*) AS frequency
FROM coal
GROUP BY Price_ZAR_USD
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_Natural_Gas, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Natural_Gas
ORDER BY frequency DESC
LIMIT 1;


SELECT Price_ICE, COUNT(*) AS frequency
FROM coal
GROUP BY Price_ICE
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_Dutch_TTF, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Dutch_TTF
ORDER BY frequency DESC
LIMIT 1;

SELECT Price_Indian_en_exg_rate, COUNT(*) AS frequency
FROM coal
GROUP BY Price_Indian_en_exg_rate
ORDER BY frequency DESC
LIMIT 1;

/* 2nd moment business decission (variance,standard deviation,range)*/
select variance(Coal_RB_4800_FOB_London_Close_USD) as var1 from coal;

select variance(Coal_RB_5500_FOB_London_Close_USD) as var2 from coal;

select variance(Coal_RB_5700_FOB_London_Close_USD) as var3 from coal;

select variance(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) as var4 from coal;

select variance(Coal_India_5500_CFR_London_Close_USD) as var5 from coal;

select variance(Price_WTI) as var6 from coal;

select variance(Price_Brent_Oil) as var7 from coal;

select variance(Price_Dubai_Brent_Oil) as var8 from coal;

select variance(Price_ExxonMobil) as var9 from coal;

select variance(Price_Shenhua) as var10 from coal;

select variance(Price_All_Share) as var11 from coal;

select variance(Price_Mining) as var12 from coal;

select variance(Price_LNG_Japan_Korea_Marker_PLATTS) as var13 from coal;

select variance(Price_ZAR_USD) as var14 from coal;

select variance(Price_Natural_Gas) as var15 from coal;

select variance(Price_ICE) as var16 from coal;

select variance(Price_Dutch_TTF) as var17 from coal;

select variance(Price_Indian_en_exg_rate) as var18 from coal;

---------------------------------standard deviation----------------------------------
select stddev(Coal_RB_4800_FOB_London_Close_USD) as sd1 from coal;
select stddev(Coal_RB_5500_FOB_London_Close_USD) as sd2 from coal;
select stddev(Coal_RB_5700_FOB_London_Close_USD) as sd3 from coal;
select stddev(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) as sd4 from coal;
select stddev(Coal_India_5500_CFR_London_Close_USD) as sd5 from coal;
select stddev(Price_WTI) as sd6 from coal;
select stddev(Price_Brent_Oil) as sd7 from coal;
select stddev(Price_Dubai_Brent_Oil) as sd8 from coal;
select stddev(Price_ExxonMobil) as sd9 from coal;
select stddev(Price_Shenhua) as sd10 from coal;
select stddev(Price_All_Share) as sd11 from coal;
select stddev(Price_Mining) as sd12 from coal;
select stddev(Price_LNG_Japan_Korea_Marker_PLATTS) as sd13 from coal;
select stddev(Price_ZAR_USD) as sd14 from coal;
select stddev(Price_Natural_Gas) as sd15 from coal;
select stddev(Price_ICE) as sd16 from coal;
select stddev(Price_Dutch_TTF) as sd17 from coal;
select stddev(Coal_RB_4800_FOB_London_Close_USD) as sd18 from coal;

----------------------------------range-----------------------------------------------

select (max(Coal_RB_4800_FOB_London_Close_USD)- min(Coal_RB_4800_FOB_London_Close_USD))
as range1 from coal;

select (max(Coal_RB_5500_FOB_London_Close_USD)- min(Coal_RB_5500_FOB_London_Close_USD))
as range2 from coal;

select (max(Coal_RB_5700_FOB_London_Close_USD)- min(Coal_RB_5700_FOB_London_Close_USD))
as range3 from coal;

select (max(Coal_RB_6000_FOB_CurrentWeek_Avg_USD)- min(Coal_RB_6000_FOB_CurrentWeek_Avg_USD))
as range4 from coal;

select (max(Coal_India_5500_CFR_London_Close_USD)- min(Coal_India_5500_CFR_London_Close_USD))
as range5 from coal;

select (max(Price_WTI)- min(Price_WTI))
as range6 from coal;

select (max(Price_Brent_Oil)- min(Price_Brent_Oil))
as range7 from coal;

select (max(Price_Dubai_Brent_Oil)- min(Price_Dubai_Brent_Oil))
as range8 from coal;

select (max(Price_ExxonMobil)- min(Price_ExxonMobil))
as range9 from coal;

select (max(Price_Shenhua)- min(Price_Shenhua))
as range10 from coal;

select (max(Price_All_Share)- min(Price_All_Share))
as range11 from coal;

select (max(Price_Mining)- min(Price_Mining))
as range12 from coal;

select (max(Price_LNG_Japan_Korea_Marker_PLATTS)- min(Price_LNG_Japan_Korea_Marker_PLATTS))
as range13 from coal;

select (max(Price_ZAR_USD)- min(Price_ZAR_USD))
as range14 from coal;

select (max(Price_Natural_Gas)- min(Price_Natural_Gas))
as range15 from coal;

select (max(Price_ICE)- min(Price_ICE))
as range16 from coal;

select (max(Price_Dutch_TTF)- min(Price_Dutch_TTF))
as range17 from coal;

select (max(Price_Indian_en_exg_rate)- min(Price_Indian_en_exg_rate))
as range18 from coal;

/*3rd(skewness)  and 4th(kurtosis) moment business decission*/

---------------------Coal_RB_4800_FOB_London_Close_USD--------------------------

SELECT
    (
        SUM(POWER(Coal_RB_4800_FOB_London_Close_USD - (SELECT AVG(Coal_RB_4800_FOB_London_Close_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_4800_FOB_London_Close_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Coal_RB_4800_FOB_London_Close_USD - (SELECT AVG(Coal_RB_4800_FOB_London_Close_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_4800_FOB_London_Close_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

--------------------------Coal_RB_5500_FOB_London_Close_USD-------------------------

SELECT
    (
        SUM(POWER(Coal_RB_5500_FOB_London_Close_USD - (SELECT AVG(Coal_RB_5500_FOB_London_Close_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_5500_FOB_London_Close_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Coal_RB_5500_FOB_London_Close_USD - (SELECT AVG(Coal_RB_5500_FOB_London_Close_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_5500_FOB_London_Close_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

------------------------------Coal_RB_5700_FOB_London_Close_USD---------------------------

SELECT
    (
        SUM(POWER(Coal_RB_5700_FOB_London_Close_USD - (SELECT AVG(Coal_RB_5700_FOB_London_Close_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_5700_FOB_London_Close_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Coal_RB_5700_FOB_London_Close_USD - (SELECT AVG(Coal_RB_5700_FOB_London_Close_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_5700_FOB_London_Close_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

------------------------------Coal_RB_6000_FOB_CurrentWeek_Avg_USD-------------------------------

SELECT
    (
        SUM(POWER(Coal_RB_6000_FOB_CurrentWeek_Avg_USD - (SELECT AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Coal_RB_6000_FOB_CurrentWeek_Avg_USD - (SELECT AVG(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_RB_6000_FOB_CurrentWeek_Avg_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

-----------------------------------Coal_India_5500_CFR_London_Close_USD------------------------------

SELECT
    (
        SUM(POWER(Coal_India_5500_CFR_London_Close_USD - (SELECT AVG(Coal_India_5500_CFR_London_Close_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_India_5500_CFR_London_Close_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Coal_India_5500_CFR_London_Close_USD - (SELECT AVG(Coal_India_5500_CFR_London_Close_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Coal_India_5500_CFR_London_Close_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

--------------------------------Price_WTI----------------------------------

SELECT
    (
        SUM(POWER(Price_WTI - (SELECT AVG(Price_WTI) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_WTI) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_WTI - (SELECT AVG(Price_WTI) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_WTI) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


----------------------------------------Price_Brent_Oil--------------------------------

SELECT
    (
        SUM(POWER(Price_Brent_Oil - (SELECT AVG(Price_Brent_Oil) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Brent_Oil) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER( - (SELECT AVG(Price_Brent_Oil) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Brent_Oil) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


-----------------------------------------Price_Dubai_Brent_Oil---------------------------------------

SELECT
    (
        SUM(POWER(Price_Dubai_Brent_Oil - (SELECT AVG(Price_Dubai_Brent_Oil) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Dubai_Brent_Oil) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Dubai_Brent_Oil - (SELECT AVG(Price_Dubai_Brent_Oil) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Dubai_Brent_Oil) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


----------------------------------------------Price_ExxonMobil--------------------------------------

SELECT
    (
        SUM(POWER(Price_ExxonMobil - (SELECT AVG(Price_ExxonMobil) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ExxonMobil) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_ExxonMobil - (SELECT AVG(Price_ExxonMobil) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ExxonMobil) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


--------------------------------------------Price_Shenhua------------------------------------------------

SELECT
    (
        SUM(POWER(Price_Shenhua - (SELECT AVG(Price_Shenhua) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Shenhua) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Shenhua - (SELECT AVG(Price_Shenhua) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Shenhua) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

----------------------------------------------------Price_All_Share---------------------------------------------

SELECT
    (
        SUM(POWER(Price_All_Share - (SELECT AVG(Price_All_Share) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_All_Share) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_All_Share - (SELECT AVG(Price_All_Share) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_All_Share) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


------------------------------------------------Price_Mining---------------------------------------------------

SELECT
    (
        SUM(POWER(Price_Mining - (SELECT AVG(Price_Mining) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Mining) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Mining - (SELECT AVG(Price_Mining) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Mining) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;


-----------------------------------------------Price_LNG_Japan_Korea_Marker_PLATTS-------------------------------------

SELECT
    (
        SUM(POWER(Price_LNG_Japan_Korea_Marker_PLATTS - (SELECT AVG(Price_LNG_Japan_Korea_Marker_PLATTS) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_LNG_Japan_Korea_Marker_PLATTS) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_LNG_Japan_Korea_Marker_PLATTS - (SELECT AVG(Price_LNG_Japan_Korea_Marker_PLATTS) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_LNG_Japan_Korea_Marker_PLATTS) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

-----------------------------------------Price_ZAR_USD-----------------------------------------------

SELECT
    (
        SUM(POWER(Price_ZAR_USD - (SELECT AVG(Price_ZAR_USD) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ZAR_USD) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_ZAR_USD - (SELECT AVG(Price_ZAR_USD) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ZAR_USD) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

--------------------------------------------Price_Natural_Gas------------------------------------------

SELECT
    (
        SUM(POWER(Price_Natural_Gas - (SELECT AVG(Price_Natural_Gas) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Natural_Gas) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Natural_Gas - (SELECT AVG(Price_Natural_Gas) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Natural_Gas) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

--------------------------------------------Price_ICE--------------------------------------------------
SELECT
    (
        SUM(POWER(Price_ICE - (SELECT AVG(Price_ICE) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ICE) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_ICE - (SELECT AVG(Price_ICE) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_ICE) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

--------------------------------------------------Price_Dutch_TTF------------------------------------------

SELECT
    (
        SUM(POWER(Price_Dutch_TTF - (SELECT AVG(Price_Dutch_TTF) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Dutch_TTF) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Dutch_TTF - (SELECT AVG(Price_Dutch_TTF) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Dutch_TTF) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;

---------------------------------------------Price_Indian_en_exg_rate--------------------------------------

SELECT
    (
        SUM(POWER(Price_Indian_en_exg_rate - (SELECT AVG(Price_Indian_en_exg_rate) FROM coal), 3)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Indian_en_exg_rate) FROM coal), 3))
    ) AS skewness,
    (
        (SUM(POWER(Price_Indian_en_exg_rate - (SELECT AVG(Price_Indian_en_exg_rate) FROM coal), 4)) / 
        (COUNT(*) * POWER((SELECT STDDEV(Price_Indian_en_exg_rate) FROM coal), 4))) - 3
    ) AS kurtosis
FROM coal;