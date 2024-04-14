import pandas as pd
import matplotlib.pyplot as plt

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Results/Results.csv"
data = pd.read_csv(path)

# Scatter plots for speed and memory usage
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.scatter(data['Operation'], data['Speed_R'], color='blue', label='R')
ax1.scatter(data['Operation'], data['Speed_Python'], color='red', label='Python')
ax1.set_ylabel('Speed (s)')
ax1.set_title('Speed Comparison Between R and Python')
ax1.legend()
ax1.set_xticklabels(data['Operation'], rotation=45) 
ax2.scatter(data['Operation'], data['Memory_R'], color='blue', label='R')
ax2.scatter(data['Operation'], data['Memory_Python'], color='red', label='Python')
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Memory Usage Comparison Between R and Python')
ax2.legend()
ax2.set_xticklabels(data['Operation'], rotation=45) 
plt.tight_layout()
plt.savefig("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Results/Results.pdf")
