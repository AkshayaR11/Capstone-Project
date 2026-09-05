from scipy import stats
# After running the SQL
severity_scores = [10, 9, 8, 6, 5, 3]  # from your distribution query
mean_priorities = [7.2, 6.8, 6.1, 5.3, 4.8, 3.1]  # corresponding means
rho, pval = stats.spearmanr(severity_scores, mean_priorities)
print(f"Spearman ρ = {rho:.2f}, p = {pval:.4f}")