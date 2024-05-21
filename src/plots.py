import pandas as pd
import matplotlib.pyplot as plt

#use of pandas to have a dataframe
df = pd.read_csv('score.csv')


column = df.iloc[:, 0]

def to_mean_vals(values, n = 50):
    for i,_ in enumerate(values):
        if i+n > len(values):
            break
        print(i, i+n)
        yield float(sum(values[i:i+n]))/n

def plot_graph(indx, vals, xlabel, ylabel, title, filename):
    plt.plot(indx, vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.savefig(filename)
    plt.show()

rel_mean_50 = list(to_mean_vals(column.values))

# Plot the graph
plot_graph(
    column.index, 
    column.values,
    'Run number', 
    'Score', 
    'Graph of Score during training', 
    'figures/score.png'
)


plot_graph(
    column.index[:-49],
    rel_mean_50, 
    'Run number', 
    'Score mean of 50 runs', 
    'Relative mean of score during training', 
    'figures/mean_50.png'
)