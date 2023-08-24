import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_matrix(array):
    # Define the column and row labels
    labels = ['F_18-22', 'F_23-45', 'F_46+']
    
    # Create a custom color map for the cell colors
    cmap = ListedColormap(['green', 'orange', 'red'])
    
    # Create a figure and set the labels for the rows and columns
    fig, ax = plt.subplots()
    
    # Remove axis ticks and labels
    ax.axis('off')
    
    # Create a table to display the matrix
    table = ax.table(cellText=array, colLabels=labels, rowLabels=labels, cellLoc='center', loc='center', cellColours=cmap(array, alpha=0.7))
    
    # Set the cell colors based on the specified conditions
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = array[i][j]
            cell = table.get_cell()[i+1, j+1]
            if value < 0.05:
                cell.set_facecolor('red')
            elif value > 0.8:
                cell.set_facecolor('orange')
            else:
                cell.set_facecolor('green')
    
    # Save the figure as a .png file
    plt.savefig('/home/ubuntu/Sexism-data-analysis/matrix.png')

# Example usage
array = [[0.1, 0.9, 0.3], [0.02, 0.6, 0.95], [0.7, 0.4, 0.1]]
create_matrix(array)
    
    


