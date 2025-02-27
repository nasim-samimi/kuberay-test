import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


font_size = 14
font_size_title = 18
font_size_legend = 14

data_cfs = pd.read_csv('normalized-response-times-cfs_cdf.txt', delimiter=r'\s+', header=None)
data_hcbs = pd.read_csv('normalized-response-times-hcbs_cdf.txt', delimiter=r'\s+', header=None)

x = data_cfs[0]  # First column (x-axis)
y = data_cfs[2]  # Third column (y-axis)

x_h = data_hcbs[0]  # First column (x-axis)
y_h = data_hcbs[2]  # Third column (y-axis)

cfs_end_x = x.iloc[-1] + 1
cfs_end_y = y.iloc[-1]

hcbs_end_x = x_h.iloc[-1] + 1
hcbs_end_y = y_h.iloc[-1]

plt.figure(figsize=(6, 4))
plt.plot(x + 1, y, linestyle='-', color='red', label='Vanilla', linewidth=3)
plt.plot(x_h + 1, y_h, linestyle='-', color='blue', label='KubeDeadline', linewidth=3)

plt.scatter([cfs_end_x], [cfs_end_y], color='red', s=100, zorder=5)
plt.scatter([hcbs_end_x], [hcbs_end_y], color='blue', s=100, zorder=5)

plt.axvline(x=cfs_end_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(x=hcbs_end_x, color='blue', linestyle='--', linewidth=2, alpha=0.7)


plt.text(cfs_end_x-0.17, cfs_end_y - 0.08, f'{cfs_end_x:.4f}', color='red', ha='center', fontsize=font_size)
plt.text(hcbs_end_x-0.17, hcbs_end_y - 0.08, f'{hcbs_end_x:.4f}', color='blue', ha='center', fontsize=font_size)

plt.xlabel('Normalised response time', fontsize=font_size)
plt.ylabel('P(Normalised response time < 1)', fontsize=font_size)
plt.legend(fontsize=font_size_legend, loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.savefig('results/cdf/cdf.svg') 
plt.close()

plt.figure(figsize=(3.5, 2))
data_cfs_s=data_cfs[0.6<(data_cfs[0]+1)]
data_cfs_s=data_cfs_s[(data_cfs_s[0]+1)<1.2]
data_hcbs_s=data_hcbs[0.6<(data_hcbs[0]+1)]
# data_hcbs_s=data_hcbs[(data_hcbs_s[0]+1)<1]
x_s=data_cfs_s[0]
y_s=data_cfs_s[2]
x_h_s=data_hcbs_s[0]
y_h_s=data_hcbs_s[2]
plt.plot(x_s+1, y_s, linestyle='-', color='red',label='Vanilla',linewidth=3,) 
plt.plot(x_h_s+1, y_h_s, linestyle='-', color='blue',label='KubeDeadline',linewidth=3,) 
# plt.xlabel('Normalised response time',fontsize=font_size)
# plt.ylabel('P(Normalised response time < 1 )',fontsize=font_size)
# plt.title('Normalised response Time',fontsize=font_size_title)
# plt.legend(fontsize=font_size_legend,loc='lower right')
plt.grid(True)
plt.savefig('results/cdf/cdf_s.svg')