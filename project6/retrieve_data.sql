-- -- For total aggregate data in US 
-- SELECT 
--     CNT as country, ST01Q01 AS grade_level,
--     AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
--     AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
--     AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
--     AVG(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_avg,
--     MAX((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_max,
--     MIN((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_min,
--     MAX((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_max,
--     MIN((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_min,
--     MAX((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_max,
--     MIN((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_min,
--     MAX(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_max,
--     MIN(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_min,
--     STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
--     STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
--     STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std,
--     STD(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_std
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1, 2;

-- -- For gender-split aggregate data in US
-- SELECT 
--     CNT as country, ST01Q01 AS grade_level, ST04Q01 AS gender,
--     AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
--     AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
--     AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
--     AVG(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_avg,
--     MAX((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_max,
--     MIN((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_min,
--     MAX((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_max,
--     MIN((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_min,
--     MAX((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_max,
--     MIN((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_min,
--     MAX(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_max,
--     MIN(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_min,
--     STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
--     STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
--     STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std,
--     STD(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_std
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1, 2, 3;

-- -- For total aggregate data everywhere else
-- SELECT 
--     CNT as country, ST01Q01 AS grade_level,
--     AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
--     AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
--     AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
--     AVG(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_avg,
--     MAX((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_max,
--     MIN((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_min,
--     MAX((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_max,
--     MIN((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_min,
--     MAX((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_max,
--     MIN((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_min,
--     MAX(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_max,
--     MIN(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_min,
--     STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
--     STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
--     STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std,
--     STD(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 +
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 +
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5
--         ) AS total_std
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1, 2;

-- -- For gender-split aggregate data everywhere else
-- SELECT 
--     CNT as country, ST01Q01 AS grade_level, ST04Q01 AS gender,
--     AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
--     AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
--     AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
--     STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
--     STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
--     STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1, 2, 3;

-- Trying something different. Get everything from the US split by gender and grade level, and binned up
-- SELECT
--     CNT as country, ST04Q01 AS gender,
--     FLOOR(
--         ((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 + 
--             (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5 +
--             (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5)
--         / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1,2,3;

-- SELECT
--     CNT as country, ST04Q01 AS gender,
--     FLOOR(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1,2,3;

-- SELECT
--     CNT as country, ST04Q01 AS gender,
--     FLOOR(
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5 / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1,2,3;

-- SELECT
--     CNT as country, ST04Q01 AS gender,
--     FLOOR(
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND (CNT IN ("United States of America", "Florida (USA)", "Massachusetts (USA)", "Connecticut (USA)"))
-- GROUP BY 1,2,3;

-- SELECT
--     CNT as country,
--     FLOOR(
--         ((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 + 
--             (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5 +
--             (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5)
--         / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1,2;

-- SELECT
--     CNT as country,
--     FLOOR(
--         (PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5 
--         / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1,2;

-- SELECT
--     CNT as country,
--     FLOOR(
--         (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5 
--         / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1,2;

-- SELECT
--     CNT as country,
--     FLOOR(
--         (PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5 / 50) * 50 AS allgrades_bucket,
--     COUNT(*) AS the_count
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1,2;

-- -- not trying to do histograms this time, just showing averages of the data
-- SELECT 
--     CNT as country,
--     AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
--     AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
--     AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
--     STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
--     STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
--     STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std
-- FROM udacity.pisa2012
-- WHERE (ST01Q01 != 96) AND (ST01Q01 != 13)
--     AND CNT != "Florida (USA)" AND CNT != "Massachusetts (USA)" AND CNT != "Connecticut (USA)"
-- GROUP BY 1;

SELECT 
    CNT AS country, ST04Q01 AS gender, COUNT(*) AS the_count,
    AVG((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_avg,
    AVG((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_avg,
    AVG((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_avg,
    STD((PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH)/5) AS math_std,
    STD((PV1READ + PV2READ + PV3READ + PV4READ + PV5READ)/5) AS read_std,
    STD((PV1SCIE + PV2SCIE + PV3SCIE + PV4SCIE + PV5SCIE)/5) AS scie_std
FROM udacity.pisa2012 
WHERE (ST01Q01 != 96) AND (ST01Q01 != 13) 
GROUP BY 1, 2;

