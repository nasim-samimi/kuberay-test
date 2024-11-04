import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import sys

if len(sys.argv) != 3:
    print("Usage: python result.py <test1> <test2>")
    sys.exit(1)

test1, test2 = sys.argv[1], sys.argv[2]

results_dir1 = f"results/{test1}-test/"
results_dir2 = f"results/{test2}-test/"

def get_results(results_dir):  # Read from response_times.csv
    data = pd.DataFrame()
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            cpu_time = int(''.join(filter(str.isdigit, name)))  # Extract integer value from filename
            df = pd.read_csv(os.path.join(results_dir, file))
            data[f"cpu_time={cpu_time}"] = df["response_time"]
    return data

# Load data for both tests
data1 = get_results(results_dir1)
data2 = get_results(results_dir2)

# Descriptive Statistics
print("Descriptive Statistics for CFS:")
print(data1.describe())
print("\nDescriptive Statistics for HCBS:")
print(data2.describe())

# Histograms
plt.figure(figsize=(12, 6))
data1.hist(bins=100, alpha=0.5, label=f"{test1}", density=True)
data2.hist(bins=100, alpha=0.5, label=f"{test2}", density=True)
plt.xlabel("Response Time")
plt.ylabel("Frequency")
plt.title("Histogram of Response Times")
plt.legend()
plt.savefig("results/histogram_comparison.png")
plt.show()

# Box Plot for Comparison
plt.figure(figsize=(10, 5))
combined_data = pd.concat([data1.melt(var_name="CFS"), data2.melt(var_name="Test2")], axis=1)
combined_data.columns = ["CPU_CFS", "Response_Time_CFS", "CPU_HCBS", "Response_Time_HCBS"]
plt.boxplot([combined_data["Response_Time_CFS"].dropna(), combined_data["Response_Time_HCBS"].dropna()],
            labels=[f"{test1}", f"{test2}"], showfliers=False)
plt.ylabel("Response Time")
plt.grid(axis='both')
plt.title("Response Time Comparison Between Scheduling Policies")
plt.savefig("results/boxplot_comparison.png")
plt.show()

# Cumulative Distribution Function (CDF)

for column in data1.columns:
    plt.figure(figsize=(10, 5))
    sorted_data1 = np.sort(data1[column].dropna())
    yvals1 = np.arange(1, len(sorted_data1) + 1) / float(len(sorted_data1))
    plt.plot(sorted_data1, yvals1, label=f"{test1} - {column}",linestyle=':')

# for column in data2.columns:
    # plt.figure(figsize=(10, 5))
    sorted_data2 = np.sort(data2[column].dropna())
    yvals2 = np.arange(1, len(sorted_data2) + 1) / float(len(sorted_data2))
    plt.plot(sorted_data2, yvals2, linestyle='--', label=f"{test2} - {column}")

    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Response Times")
    plt.legend()
    plt.savefig(f"results/cdf_comparison_{column}.png")
    plt.show()

# Statistical Tests
# Flatten the data for each test for global comparison
data1_flat = data1.values.flatten()
data2_flat = data2.values.flatten()

# T-test and Mann-Whitney U test
t_stat, t_pval = ttest_ind(data1_flat, data2_flat, nan_policy='omit')
mwu_stat, mwu_pval = mannwhitneyu(data1_flat, data2_flat, alternative="two-sided")

print(f"T-test p-value: {t_pval}")
print(f"Mann-Whitney U test p-value: {mwu_pval}")

# Save summary statistics
summary_stats = pd.DataFrame({
    f"{test1} Mean": [data1_flat.mean()],
    f"{test2} Mean": [data2_flat.mean()],
    f"{test1} Median": [np.median(data1_flat)],
    f"{test2} Median": [np.median(data2_flat)],
    f"{test1} Std Dev": [data1_flat.std()],
    f"{test2} Std Dev": [data2_flat.std()],
    "T-test p-value": [t_pval],
    "Mann-Whitney U p-value": [mwu_pval]
})
summary_stats.to_csv("summary_statistics.csv", index=False)
print("\nSummary statistics saved as summary_statistics.csv")
