import matplotlib.pyplot as plt

def plot_training_history(history):
    # List of metrics to plot
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score']
    
    # Create a figure with subplots
    fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    fig.suptitle('Model Training History')
    
    for i, metric in enumerate(metrics):
        # Plot training & validation metric values
        axs[i].plot(history.history[metric], label=f'Train {metric}')
        axs[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        
        axs[i].set_title(f'{metric.capitalize()} Over Epochs')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()