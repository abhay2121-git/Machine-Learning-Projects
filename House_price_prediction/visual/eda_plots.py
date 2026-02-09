"""
Exploratory Data Analysis module for house price prediction.
Contains functions for creating insightful visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


def plot_feature_distributions(data: pd.DataFrame, target: pd.Series) -> None:
    """
    Create distribution plots for all features.
    
    Args:
        data: Feature DataFrame
        target: Target variable (house prices)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 12))
    fig.suptitle('Feature Distributions and Price Distribution', fontsize=16, fontweight='bold')
    
    # Plot distributions for each feature
    for i, (feature, ax) in enumerate(zip(data.columns, axes.flatten())):
        # Histogram for numerical features
        if data[feature].dtype in ['int64', 'float64']:
            ax.hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Box plot for categorical features
        elif data[feature].dtype == 'object':
            ax.boxplot(x=data[feature], y=target, ax=ax)
            ax.set_title(f'{feature} vs Price')
            ax.set_xlabel(feature)
            ax.set_ylabel('Price ($100k)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """
    Create correlation matrix heatmap.
    
    Args:
        data: Feature DataFrame
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=True,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_geographical_distribution(data: pd.DataFrame, target: pd.Series) -> None:
    """
    Create geographical scatter plot of house prices.
    
    Args:
        data: Feature DataFrame with longitude and latitude
        target: House prices
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(data['Longitude'], data['Latitude'], 
                           c=data['MedInc'], s=data['Population'],
                           cmap='viridis', alpha=0.6)
    
    plt.colorbar(scatter, label='Median Income')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Distribution of House Prices')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_price_vs_features(data: pd.DataFrame, target: pd.Series) -> None:
    """
    Create plots showing relationship between price and key features.
    
    Args:
        data: Feature DataFrame
        target: House prices
    """
    # Select key features for plotting
    key_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('House Price vs Key Features', fontsize=16, fontweight='bold')
    
    for i, (feature, ax) in enumerate(zip(key_features, axes.flatten())):
        ax.scatter(data[feature], target, alpha=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel('Price ($100k)')
        ax.set_title(f'Price vs {feature}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
