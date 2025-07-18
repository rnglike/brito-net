import os
import numpy as np
import matplotlib.pyplot as plt

def plot_against_epochs(args, name):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    loss_history, accuracy_history = args
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, label='Loss (MSE)', color='blue')
    plt.plot(accuracy_history, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy (' + name + ')')
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/training_results_{name}.png')
    print(f"Plot saved as 'plots/training_results_{name}.png'")