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

font_size = 16

def get_results(results_dir):  # Read from response_times.csv
    data = pd.DataFrame()
    all_data=pd.DataFrame()
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            cpu_time = int(''.join(filter(str.isdigit, name)))  # Extract integer value from filename
            df = pd.read_csv(os.path.join(results_dir, file))
            data[f"{cpu_time}%"] = df["response_time"]
            all_data = pd.concat([all_data, df["response_time"]], axis=0)
    return data, all_data

# Load data for both tests
data1,test1_all_data = get_results(results_dir1)
data2,test2_all_data = get_results(results_dir2)

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

# CDF Calculation and Plotting for Combined Data
def cdf_plot(test1_all_data, test2_all_data, test1, test2):
    plt.figure(figsize=(10, 5))
    sorted_data1 = np.sort(test1_all_data["response_time"].dropna()) 
    print(sorted_data1)
    yvals1 = np.arange(1, len(sorted_data1) + 1) / float(len(sorted_data1))  # Calculate CDF values
    plt.plot(sorted_data1, yvals1, label=f"{test1}", linestyle=':')

    sorted_data2 = np.sort(test2_all_data["response_time"].dropna())  # Sort data for second dataset
    print(sorted_data2)
    yvals2 = np.arange(1, len(sorted_data2) + 1) / float(len(sorted_data2))  # Calculate CDF values
    plt.plot(sorted_data2, yvals2, linestyle='--', label=f"{test2}")

    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Response Times")
    plt.legend()
    plt.savefig("results/cdf_comparison_combined.png")
# plt.show()

# Statistical Tests
# Flatten the data for each test for global comparison
data1_flat = data1.values.flatten()
data2_flat = data2.values.flatten()

def plot_max_fps_comparison(test1, test2, data1, data2):
    # Calculate maximum FPS for each column in data1 and data2
    max_fps1 = 1 / data1.max()  # Calculate maximum FPS for test1
    max_fps2 = 1 / data2.max()  # Calculate maximum FPS for test2

    red_color = '#d62d20'  
    blue_color = '#0057e7'

    max_fps1 = max_fps1.replace([np.inf, -np.inf], np.nan).fillna(0)
    max_fps2 = max_fps2.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Create a plot showing both maximum FPS values for each column
    plt.figure(figsize=(8, 5))
    width = 0.35  # Width of the bars
    x = np.arange(len(max_fps1.index))  # X-axis positions for data1 columns

    # Plot bars for data1 and data2
    plt.bar(x - width/2, max_fps1, width, label=f"{test1} - Max FPS",  alpha=0.4, edgecolor='black')
    plt.bar(x + width/2, max_fps2, width, label=f"{test2} - Max FPS",  alpha=0.4, edgecolor='black')

    # Labeling the graph
    plt.ylabel("Maximum FPS")
    plt.xlabel("CPU Utilisation")
    plt.title(f"Maximum FPS Comparison for {test1} and {test2}")
    plt.xticks(x, max_fps1.index)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("results/max_fps_comparison.png")


# Plot for both tests
plot_max_fps_comparison(test1, test2, data1, data2)

# cdf_plot(test1_all_data, test2_all_data, test1, test2)
