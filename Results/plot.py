import pandas as pd
import matplotlib.pyplot as plt

# Load data
path = "/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Results/Results.csv"
data = pd.read_csv(path)

# Setting up the figure and axes for the subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Scatter plot for Speed Comparison
ax1.scatter(data['Operation'], data['Speed_R'], color='blue', label='R')
ax1.scatter(data['Operation'], data['Speed_Python'], color='red', label='Python')
ax1.set_ylabel('Speed (s)')
ax1.set_title('Speed Comparison Between R and Python')
ax1.legend()
ax1.set_xticklabels(data['Operation'], rotation=45) 

# Scatter plot for Memory Usage Comparison
ax2.scatter(data['Operation'], data['Memory_R'], color='blue', label='R')
ax2.scatter(data['Operation'], data['Memory_Python'], color='red', label='Python')
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Memory Usage Comparison Between R and Python')
ax2.legend()
ax2.set_xticklabels(data['Operation'], rotation=45) 

# Improve layout and display the plots
plt.tight_layout()
plt.savefig("/Users/ancavanciupopescu/Desktop/Classes/CS 32/Final project/Results/Results.pdf")
