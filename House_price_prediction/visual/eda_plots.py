"""
Exploratory Data Analysis (EDA) plotting utilities for house price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_distribution(df: pd.DataFrame, column: str, figsize: tuple = (10, 6)) -> None:
    """
    Plot distribution of a numerical column.
    
    Args:
        df: DataFrame
        column: Column name to plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    sns.histplot(df[column], kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')
    
    # Box plot
    sns.boxplot(y=df[column], ax=ax2)
    ax2.set_title(f'Box Plot of {column}')
    ax2.set_ylabel(column)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame
        figsize: Figure size
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                hue: Optional[str] = None, figsize: tuple = (10, 6)) -> None:
    """
    Plot scatter plot between two numerical columns.
    
    Args:
        df: DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        hue: Column for color coding (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.6)
    plt.title(f'{y_col} vs {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_categorical_analysis(df: pd.DataFrame, cat_col: str, 
                            target_col: str, figsize: tuple = (12, 6)) -> None:
    """
    Plot categorical variable analysis against target.
    
    Args:
        df: DataFrame
        cat_col: Categorical column
        target_col: Target numerical column
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    df[cat_col].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title(f'Distribution of {cat_col}')
    ax1.set_xlabel(cat_col)
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Box plot by category
    sns.boxplot(data=df, x=cat_col, y=target_col, ax=ax2)
    ax2.set_title(f'{target_col} by {cat_col}')
    ax2.set_xlabel(cat_col)
    ax2.set_ylabel(target_col)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_missing_values(df: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """
    Plot missing values heatmap.
    
    Args:
        df: DataFrame
        figsize: Figure size
    """
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        print("No missing values found!")
        return
    
    plt.figure(figsize=figsize)
    sns.barplot(x=missing_data.index, y=missing_data.values)
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_eda_report(df: pd.DataFrame, target_column: str) -> None:
    """
    Generate comprehensive EDA report.
    
    Args:
        df: DataFrame
        target_column: Target column name
    """
    print("=== EDA Report ===")
    print(f"Dataset Shape: {df.shape}")
    print(f"Target Column: {target_column}")
    print("\n")
    
    # Target distribution
    plot_distribution(df, target_column)
    
    # Correlation matrix
    plot_correlation_matrix(df)
    
    # Missing values
    plot_missing_values(df)
    
    # Top correlations with target
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_column].sort_values(ascending=False)
    
    print("Top correlations with target:")
    print(correlations.head(10))


if __name__ == "__main__":
    # Example usage
    # df = load_data('data/house_prices.csv')
    # generate_eda_report(df, 'price')
    print("EDA plotting utilities loaded successfully!")
