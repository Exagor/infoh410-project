import pandas as pd
import matplotlib.pyplot as plt

#use of pandas to have a dataframe
df = pd.read_csv('score.csv')
#select the first column
column = df.iloc[:, 0]


def to_mean_vals(values, n = 50):
    """Calculate the mean of the last n values"""
    for i,_ in enumerate(values):
        if i+n > len(values):
            break
        print(i, i+n)
        yield float(sum(values[i:i+n]))/n

def plot_graph(indx, vals, xlabel, ylabel, title, filename):
    """Plot a graph of the given values"""
    plt.plot(indx, vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    plt.savefig(filename)
    plt.show()


# Raw score graph
plot_graph(
    column.index, 
    column.values,
    'Run number', 
    'Score', 
    'Graph of Score during training', 
    'figures/score.png'
)

# Relative mean (n=50) graph
rel_mean_50 = list(to_mean_vals(column.values))

plot_graph(
    column.index[:-49],
    rel_mean_50, 
    'Run number', 
    'Score mean of 50 runs', 
    'Relative mean of score during training', 
    'figures/mean_50.png'
)