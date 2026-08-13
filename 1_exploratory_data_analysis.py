import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # 1. Setup directories and load data
    os.makedirs('visuals', exist_ok=True)
    df = pd.read_csv('cleaned_psychiatric_data.csv')
    
    # 2. Missing Value Graph
    print("Generating Missing Value Graph...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values in Dataset')
    plt.tight_layout()
    plt.savefig('visuals/1_missing_values.png', dpi=300)
    plt.close()
    
    # 3. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(16, 12))
    # Using annot=False if too many features, but annot=True with smaller font is good
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('visuals/2_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Class Distribution Histogram
    print("Generating Class Distribution Histogram...")
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df, 
        x='Mental_Health_Status', 
        palette='Set2', 
        hue='Mental_Health_Status', 
        legend=False
    )
    plt.title('Class Distribution of Mental Health Status')
    plt.xlabel('Mental Health Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visuals/3_class_distribution.png', dpi=300)
    plt.close()
    
    # 5. Box Plots
    print("Generating Box Plots...")
    features = numeric_df.columns
    n_features = len(features)
    cols = 4
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(
            x='Mental_Health_Status', 
            y=feature, 
            data=df, 
            ax=axes[i], 
            hue='Mental_Health_Status', 
            palette='Set3',
            legend=False
        )
        axes[i].set_title(f'{feature} vs Mental Health Status')
        axes[i].tick_params(axis='x', rotation=45)
        
    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    fig.savefig('visuals/4_feature_boxplots.png', dpi=300)
    plt.close()

    print("EDA Visuals generated successfully in the 'visuals' directory.")

if __name__ == "__main__":
    main()
