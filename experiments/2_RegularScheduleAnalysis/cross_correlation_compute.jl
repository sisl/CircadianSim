# Spearman correlation matrices for regular-schedule particle summaries.
# Used by computeCrossCorrelations.jl and plotCorrelationsAndSHAPCombined.jl.
# Requires: StatsBase (corspearman), globals HAS_HYPOTHESISTESTS, HAS_DISTRIBUTIONS,
#           HypothesisTests (optional), Distributions (optional).

function compute_correlation(vec1, vec2)
    valid_indices = Int[]
    for idx in eachindex(vec1)
        if idx <= length(vec2) && isfinite(vec1[idx]) && isfinite(vec2[idx])
            push!(valid_indices, idx)
        end
    end

    if length(valid_indices) < 3
        return NaN, NaN
    end

    vec1_clean = [vec1[idx] for idx in valid_indices]
    vec2_clean = [vec2[idx] for idx in valid_indices]

    corr_val = corspearman(vec1_clean, vec2_clean)

    p_val = NaN
    n = length(vec1_clean)

    if HAS_HYPOTHESISTESTS
        try
            test_result = SpearmanRankCorrelationTest(vec1_clean, vec2_clean)
            p_val = pvalue(test_result)
            if isnan(p_val) || isinf(p_val)
                p_val = NaN
            end
        catch
            p_val = NaN
        end
    end

    if isnan(p_val) && n >= 3 && abs(corr_val) < 1.0
        try
            t_stat = corr_val * sqrt((n - 2) / (1 - corr_val^2))

            if HAS_DISTRIBUTIONS
                t_dist = TDist(n - 2)
                p_val = 2 * (1 - cdf(t_dist, abs(t_stat)))
            else
                if n > 30
                    z_abs = abs(t_stat)
                    if z_abs > 3.0
                        p_val = 0.001
                    elseif z_abs > 2.0
                        p_val = 0.05
                    elseif z_abs > 1.96
                        p_val = 0.05
                    else
                        p_val = 0.1
                    end
                else
                    if abs(corr_val) > 0.8 && n > 10
                        p_val = 0.001
                    elseif abs(corr_val) > 0.6 && n > 10
                        p_val = 0.01
                    else
                        p_val = NaN
                    end
                end
            end

            if isnan(p_val) || isinf(p_val) || p_val < 0 || p_val > 1
                p_val = NaN
            end
        catch
            p_val = NaN
        end
    elseif n < 3
        p_val = NaN
    elseif abs(corr_val) >= 1.0
        p_val = 0.0
    end

    return corr_val, p_val
end

function compute_spearman_correlation_matrices(input_data, output_data, n_inputs::Int, n_outputs::Int)
    correlation_matrix = zeros(n_inputs, n_outputs)
    pvalue_matrix = zeros(n_inputs, n_outputs)
    output_correlation_matrix = zeros(n_outputs, n_outputs)
    output_pvalue_matrix = zeros(n_outputs, n_outputs)

    for i in 1:n_inputs
        for j in 1:n_outputs
            corr_val, p_val = compute_correlation(input_data[i], output_data[j])
            correlation_matrix[i, j] = corr_val
            pvalue_matrix[i, j] = p_val
        end
    end

    for i in 1:n_outputs
        for j in 1:n_outputs
            if i == j
                output_correlation_matrix[i, j] = 1.0
                output_pvalue_matrix[i, j] = 0.0
            else
                corr_val, p_val = compute_correlation(output_data[i], output_data[j])
                output_correlation_matrix[i, j] = corr_val
                output_pvalue_matrix[i, j] = p_val
            end
        end
    end

    return correlation_matrix, pvalue_matrix, output_correlation_matrix, output_pvalue_matrix
end
