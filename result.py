import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import sys

# if len(sys.argv) != 3:
#     print("Usage: python result.py <test1> <test2>")
#     sys.exit(1)

main_dir="results/efficientnet/"

results_dir1 = f"{main_dir}/cfs-test/"
results_dir2 = f"{main_dir}/hcbs-test/"

test1="Vanilla"
test2="KubeDeadline"

font_size = 14
font_size_title = 18
font_size_legend = 14
font_size_ticks = 11
figsize=(6,4)

def get_results(results_dir): 
    data = pd.DataFrame()
    all_data=pd.DataFrame()
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            cpu_time = int(''.join(filter(str.isdigit, name)))  
            df = pd.read_csv(os.path.join(results_dir, file))
            data[f"{cpu_time}%"] = df["response_time"]
            all_data = pd.concat([all_data, df["response_time"]], axis=0)
    return data, all_data


data1,test1_all_data = get_results(results_dir1)
data2,test2_all_data = get_results(results_dir2)

print("Descriptive Statistics for Vanilla:")
print(data1.describe())
print("\nDescriptive Statistics for KubeDeadline:")
print(data2.describe())


plt.figure(figsize=figsize)
combined_data = pd.concat([data1.melt(var_name="CFS"), data2.melt(var_name="Test2")], axis=1)
combined_data.columns = ["Vanilla", "Response_Time_Vanilla", "CPU_KubeDeadline", "Response_Time_KubeDeadline"]
plt.boxplot([combined_data["Response_Time_Vanilla"].dropna(), combined_data["Response_Time_KubeDeadline"].dropna()],
            labels=[f"{test1}", f"{test2}"], showfliers=False)
plt.ylabel("Response Time")
plt.grid(axis='both')
plt.title("Response Time Comparison Between Scheduling Policies")
# plt.savefig(f"{main_dir}/boxplot_comparison.pdf")



for column in data1.columns:
    plt.figure(figsize=(8,5))
    sorted_data1 = np.sort(data1[column].dropna())
    yvals1 = np.arange(1, len(sorted_data1) + 1) / float(len(sorted_data1))
    plt.plot(sorted_data1, yvals1, label=f"{test1} - {column}",linestyle=':')

    sorted_data2 = np.sort(data2[column].dropna())
    yvals2 = np.arange(1, len(sorted_data2) + 1) / float(len(sorted_data2))
    plt.plot(sorted_data2, yvals2, linestyle='--', label=f"{test2} - {column}")

    plt.xlabel("Response Time")
    plt.ylabel("Cumulative Probability")
    # plt.title("CDF of Response Times")
    plt.legend()
    # plt.savefig(f"{main_dir}/cdf_comparison_{column}.pdf")

# CDF Calculation and Plotting for Combined Data
def cdf_plot(test1_all_data, test2_all_data, test1, test2):
    plt.figure(figsize=(8,5))
    sorted_data1 = np.sort(test1_all_data["response_time"].dropna()) 
    print(sorted_data1)
    yvals1 = np.arange(1, len(sorted_data1) + 1) / float(len(sorted_data1))  
    plt.plot(sorted_data1, yvals1, label=f"{test1}", linestyle='-',linewidth=3,color='purple')

    sorted_data2 = np.sort(test2_all_data["response_time"].dropna()) 
    print(sorted_data2)
    yvals2 = np.arange(1, len(sorted_data2) + 1) / float(len(sorted_data2))
    plt.plot(sorted_data2, yvals2, linestyle='solid', label=f"{test2}",linewidth=3,color='green')

    plt.xlabel("Response Time",fontsize=font_size)
    plt.ylabel("Cumulative Probability",fontsize=font_size)
    # plt.title("CDF of Response Times",fontsize=font_size_title)
    plt.legend(fontsize=font_size_legend, loc='lower right')
    plt.grid()
    plt.savefig(f"{main_dir}/cdf_comparison_combined.pdf")

# plt.show()

data1_flat = data1.values.flatten()
data2_flat = data2.values.flatten()

def plot_max_fps_comparison(test1, test2, data1, data2):
    max_fps1 = 1 / data1.max()  # Calculate maximum FPS for test1
    max_fps2 = 1 / data2.max()  # Calculate maximum FPS for test2


    max_fps1 = max_fps1.replace([np.inf, -np.inf], np.nan).fillna(0)
    max_fps2 = max_fps2.replace([np.inf, -np.inf], np.nan).fillna(0)

    plt.figure(figsize=figsize)
    width = 0.35  # Width of the bars
    x = np.arange(len(max_fps1.index))  # X-axis positions for data1 columns

    plt.bar(x - width/2, max_fps1, width, label=f"{test1} - Min FPS", hatch="//" , edgecolor='black')
    plt.bar(x + width/2, max_fps2, width, label=f"{test2} - Min FPS", hatch="\\" , edgecolor='black')

    plt.ylabel("Minimum FPS",fontsize=font_size)
    plt.xlabel("CPU Utilisation",fontsize=font_size)
    # plt.title(f"Minimum FPS for {test1} and {test2}",fontsize=font_size_title)
    plt.yticks(fontsize=font_size_ticks)
    plt.xticks(x, max_fps1.index,fontsize=font_size_ticks)
    plt.legend(fontsize=font_size_legend,loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{main_dir}/min_fps_comparison.pdf")


# Plot for both tests
plot_max_fps_comparison(test1, test2, data1, data2)

cdf_plot(test1_all_data, test2_all_data, test1, test2)

def plot_avg_fps_comparison(test1, test2, data1, data2):
    avg_fps1 = 1 / data1.mean()  # Calculate maximum FPS for test1
    avg_fps2 = 1 / data2.mean()  # Calculate maximum FPS for test2

    red_color = '#d62d20'  
    blue_color = '#0057e7'

    avg_fps1 = avg_fps1.replace([np.inf, -np.inf], np.nan).fillna(0)
    avg_fps2 = avg_fps2.replace([np.inf, -np.inf], np.nan).fillna(0)

    plt.figure(figsize=figsize)
    width = 0.35  # Width of the bars
    x = np.arange(len(avg_fps1.index)) 

    # Plot bars for data1 and data2
    plt.bar(x - width/2, avg_fps1, width, label=f"{test1} - avg FPS", hatch="//" , edgecolor='black')
    plt.bar(x + width/2, avg_fps2, width, label=f"{test2} - avg FPS", hatch="\\" , edgecolor='black')

    # plt.plot(x - width/2, avg_fps1, width, label=f"{test1} - avg FPS", hatch="//" , edgecolor='black')
    # plt.plot(x + width/2, avg_fps2, width, label=f"{test2} - avg FPS", hatch="\\" , edgecolor='black')

    plt.ylabel("Average FPS",fontsize=font_size)
    plt.xlabel("CPU Utilisation",fontsize=font_size)
    # plt.title(f"Average FPS for {test1} and {test2}",fontsize=font_size_title)
    plt.xticks(x, avg_fps1.index)
    plt.legend(fontsize=font_size_legend, loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{main_dir}/avg_fps_comparison.pdf")

plot_avg_fps_comparison(test1, test2, data1, data2)

import matplotlib.pyplot as plt
import numpy as np

def plot_avg_fps_comparison_line(test1, test2, data1, data2):
    avg_fps1 = 1 / data1.mean()  # Calculate average FPS for test1
    avg_fps2 = 1 / data2.mean()  # Calculate average FPS for test2

    # Handle infinite and NaN values
    avg_fps1 = avg_fps1.replace([np.inf, -np.inf], np.nan).fillna(0)
    avg_fps2 = avg_fps2.replace([np.inf, -np.inf], np.nan).fillna(0)

    x = np.arange(len(avg_fps1.index))

    plt.figure(figsize=figsize)
    plt.plot(x, avg_fps1, marker='*', linestyle='-', color='red', label=f"{test1} - avg FPS")
    plt.plot(x, avg_fps2, marker='o', linestyle='-', color='blue', label=f"{test2} - avg FPS")

    # Labeling the graph
    plt.ylabel("Average FPS", fontsize=font_size)
    plt.xlabel("CPU Utilisation", fontsize=font_size)
    plt.title(f"Average FPS for {test1} and {test2}", fontsize=font_size_title)
    plt.xticks(x, avg_fps1.index, fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.legend(fontsize=font_size_legend, loc='lower right')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{main_dir}/avg_fps_comparison_line.pdf")

# plot_avg_fps_comparison_line("KubeDeadline", "Vanilla", data1, data2)


def plot_max_fps_comparison_line(test1, test2, data1, data2):
    max_fps1 = 1 / data1.max()  # Calculate average FPS for test1
    max_fps2 = 1 / data2.max()  # Calculate average FPS for test2

    max_fps1 = max_fps1.replace([np.inf, -np.inf], np.nan).fillna(0)
    max_fps2 = max_fps2.replace([np.inf, -np.inf], np.nan).fillna(0)

    x = np.arange(len(max_fps1.index))

    plt.figure(figsize=figsize)
    plt.plot(x, max_fps1, marker='*', linestyle='-', color='red', label=f"{test1} - max FPS")
    plt.plot(x, max_fps2, marker='o', linestyle='-', color='blue', label=f"{test2} - max FPS")

    plt.ylabel("Average FPS", fontsize=font_size)
    plt.xlabel("CPU Utilisation", fontsize=font_size)
    plt.title(f"Average FPS for {test1} and {test2}", fontsize=font_size_title)
    plt.xticks(x, max_fps1.index)
    plt.legend(fontsize=font_size_legend, loc='lower right')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{main_dir}/max_fps_comparison_line.pdf")

# plot_max_fps_comparison_line("KubeDeadline", "Vanilla", data1, data2)