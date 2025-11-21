"""
Model Performance Visualization Script
=======================================
This script generates various visualizations and metrics for AI model evaluation.

Required Libraries:
pip install matplotlib seaborn scikit-learn numpy pandas
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def generate_dummy_data():
    """
    Generate dummy training data for demonstration purposes.
    Replace this with actual model training results.
    
    Returns:
        tuple: (train_accuracy, train_loss, true_labels, predicted_labels)
    """
    # Number of epochs
    num_epochs = 50
    
    # Simulate training accuracy (improving over time)
    train_accuracy = np.linspace(0.65, 0.95, num_epochs) + np.random.normal(0, 0.02, num_epochs)
    train_accuracy = np.clip(train_accuracy, 0, 1)  # Keep values between 0 and 1
    
    # Simulate training loss (decreasing over time)
    train_loss = np.exp(-np.linspace(0, 3, num_epochs)) + np.random.normal(0, 0.05, num_epochs)
    train_loss = np.clip(train_loss, 0.05, None)  # Keep positive values
    
    # Number of samples for predictions
    num_samples = 1000
    
    # Simulate true labels (5 classes: 0-4)
    true_labels = np.random.randint(0, 5, num_samples)
    
    # Simulate predicted labels with ~85% accuracy
    predicted_labels = true_labels.copy()
    # Randomly change 15% of predictions
    num_errors = int(0.15 * num_samples)
    error_indices = np.random.choice(num_samples, num_errors, replace=False)
    for idx in error_indices:
        # Pick a different label
        wrong_label = np.random.choice([i for i in range(5) if i != true_labels[idx]])
        predicted_labels[idx] = wrong_label
    
    return train_accuracy, train_loss, true_labels, predicted_labels


def plot_accuracy_curve(train_accuracy, save_path='accuracy_curve.png'):
    """
    Generate and save accuracy curve plot.
    
    Args:
        train_accuracy (array): Training accuracy values per epoch
        save_path (str): Path to save the plot
    """
    epochs = np.arange(1, len(train_accuracy) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, linewidth=2, color='#2E86AB', marker='o', 
             markersize=4, markevery=5, label='Training Accuracy')
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='lower right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy curve saved to: {save_path}")
    plt.close()


def plot_loss_curve(train_loss, save_path='loss_curve.png'):
    """
    Generate and save loss curve plot.
    
    Args:
        train_loss (array): Training loss values per epoch
        save_path (str): Path to save the plot
    """
    epochs = np.arange(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, linewidth=2, color='#A23B72', marker='s', 
             markersize=4, markevery=5, label='Training Loss')
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Loss curve saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(true_labels, predicted_labels, save_path='confusion_matrix.png'):
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        true_labels (array): True class labels
        predicted_labels (array): Predicted class labels
        save_path (str): Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create class labels
    classes = [f'Class {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate and print classification metrics.
    
    Args:
        true_labels (array): True class labels
        predicted_labels (array): Predicted class labels
    """
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    # Print formatted metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("="*50 + "\n")
    
    return accuracy, precision, recall, f1


def main():
    """
    Main function to execute the entire visualization pipeline.
    """
    print("\n" + "="*50)
    print("AI MODEL VISUALIZATION SCRIPT")
    print("="*50 + "\n")
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Generate or load data
    print("Loading model data...")
    train_accuracy, train_loss, true_labels, predicted_labels = generate_dummy_data()
    print("✓ Data loaded successfully\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_accuracy_curve(train_accuracy)
    plot_loss_curve(train_loss)
    plot_confusion_matrix(true_labels, predicted_labels)
    
    # Calculate and display metrics
    calculate_metrics(true_labels, predicted_labels)
    
    print("="*50)
    print("Graph generation completed!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
