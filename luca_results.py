import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font_size = 14
font_size_title = 18
font_size_legend = 14

data_cfs = pd.read_csv('normalized-response-times-cfs_cdf.txt', delimiter=r'\s+', header=None) 
data_hcbs = pd.read_csv('normalized-response-times-hcbs_cdf.txt', delimiter=r'\s+', header=None) 

print(data_cfs.head())
print(data_cfs.columns)


x = data_cfs[0]  # First column (x-axis)
y = data_cfs[2]  # Third column (y-axis)

x_h = data_hcbs[0]  # First column (x-axis)
y_h = data_hcbs[2]  # Third column (y-axis)
# Plotting the data
plt.figure(figsize=(6, 4))
plt.plot(x+1, y, linestyle='-', color='red',label='Vanilla',linewidth=3,) 
plt.plot(x_h+1, y_h, linestyle='-', color='blue',label='KubeDeadline',linewidth=3,) 
plt.xlabel('Normalised response time',fontsize=font_size)
plt.ylabel('P(Normalised response time < 1 )',fontsize=font_size)
# plt.title('Normalised response Time',fontsize=font_size_title)
plt.legend(fontsize=font_size_legend,loc='lower right')
plt.grid(True)

plt.savefig('cdf.pdf') 
plt.close()

# plt.figure(figsize=(3.5, 2))
# data_cfs_s=data_cfs[0.4<(data_cfs[0]+1)]
# # data_cfs_s=data_cfs[(data_cfs_s[0]+1)<1]
# data_hcbs_s=data_hcbs[0.4<(data_hcbs[0]+1)]
# # data_hcbs_s=data_hcbs[(data_hcbs_s[0]+1)<1]
# x_s=data_cfs_s[0]
# y_s=data_cfs_s[2]
# x_h_s=data_hcbs_s[0]
# y_h_s=data_hcbs_s[2]
# plt.plot(x_s+1, y_s, linestyle='-', color='red',label='Vanilla',linewidth=3,) 
# plt.plot(x_h_s+1, y_h_s, linestyle='-', color='blue',label='KubeDeadline',linewidth=3,) 
# # plt.xlabel('Normalised response time',fontsize=font_size)
# # plt.ylabel('P(Normalised response time < 1 )',fontsize=font_size)
# # plt.title('Normalised response Time',fontsize=font_size_title)
# # plt.legend(fontsize=font_size_legend,loc='lower right')
# plt.grid(True)
# plt.savefig('cdf_s.pdf')