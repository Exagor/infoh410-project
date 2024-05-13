import pandas as pd
import matplotlib.pyplot as plt

#use of pandas to have a dataframe
df = pd.read_csv('score.csv')


column = df.iloc[:, 0]

# Plot the graph
plt.plot(column.index, column.values)
plt.xlabel('Run number')
plt.ylabel('Score')
plt.title('Graph of Score during training')
plt.grid(True)

plt.savefig("figures/score.png")
plt.show()
