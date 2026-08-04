-- Query 1: Selected pairs for a supplied run_id
SELECT
    run_id,
    formation_start,
    formation_end,
    symbol_y,
    symbol_x,
    group_name,
    observations,
    alpha,
    beta,
    spread_standard_deviation,
    cointegration_statistic,
    cointegration_pvalue,
    corrected_pvalue,
    adf_statistic,
    adf_pvalue,
    half_life,
    hurst,
    selected,
    rank,
    rejection_reasons,
    loaded_at
FROM pair_screening_results
WHERE run_id = $run_id
  AND selected
ORDER BY
    rank ASC,
    corrected_pvalue ASC NULLS LAST,
    symbol_y ASC,
    symbol_x ASC;

-- Query 2: Selection counts and rates by group_name
SELECT
    group_name,
    COUNT(*) AS total_pairs,
    COUNT(*) FILTER (WHERE selected) AS selected_pairs,
    CAST(COUNT(*) FILTER (WHERE selected) AS DOUBLE)
        / COUNT(*) AS selection_rate
FROM pair_screening_results
GROUP BY group_name
ORDER BY
    selection_rate DESC,
    selected_pairs DESC,
    total_pairs DESC,
    group_name ASC NULLS LAST;

-- Query 3: Most frequently selected canonical pairs across runs
SELECT
    symbol_y,
    symbol_x,
    COUNT(DISTINCT run_id) AS selected_run_count
FROM pair_screening_results
WHERE selected
GROUP BY symbol_y, symbol_x
ORDER BY
    selected_run_count DESC,
    symbol_y ASC,
    symbol_x ASC;

-- Query 4: Average selected-pair diagnostics across runs
SELECT
    symbol_y,
    symbol_x,
    COUNT(DISTINCT run_id) AS selected_run_count,
    AVG(cointegration_pvalue) AS mean_cointegration_pvalue,
    AVG(corrected_pvalue) AS mean_corrected_pvalue,
    AVG(adf_pvalue) AS mean_adf_pvalue,
    AVG(half_life) AS mean_half_life,
    AVG(hurst) AS mean_hurst
FROM pair_screening_results
WHERE selected
GROUP BY symbol_y, symbol_x
ORDER BY
    selected_run_count DESC,
    symbol_y ASC,
    symbol_x ASC;

-- Query 5: Rejection-reason counts
SELECT
    json_extract_string(reason.value, '$') AS rejection_reason,
    COUNT(*) AS rejection_count
FROM pair_screening_results AS result,
LATERAL json_each(CAST(result.rejection_reasons AS JSON)) AS reason
WHERE NOT result.selected
GROUP BY rejection_reason
ORDER BY
    rejection_count DESC,
    rejection_reason ASC;
