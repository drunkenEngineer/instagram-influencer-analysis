import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import re

# Set the style for the visualizations
plt.style.use('fivethirtyeight')
sns.set(style='whitegrid')

# Load the data
print("Loading Instagram influencer data...")
file_path = 'social media influencers-INSTAGRAM - -DEC 2022.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Data preprocessing
def preprocess_data(df):
    print("Preprocessing data...")
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Rename columns to remove spaces and special characters
    data.columns = [col.strip() for col in data.columns]
    
    # Convert follower counts from string (with M/K) to numeric values
    def convert_to_numeric(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip()
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000
        else:
            try:
                return float(value)
            except:
                return np.nan
    
    # Apply conversion to follower and engagement columns
    for col in ['followers', 'Eng. (Auth.)', 'Eng. (Avg.)']:
        data[col] = data[col].apply(convert_to_numeric)
    
    # Calculate engagement rate (Avg. Engagement / Followers * 100)
    data['engagement_rate'] = (data['Eng. (Avg.)'] / data['followers']) * 100
    
    # Fill missing values in Category columns with 'Unknown'
    data['Category_1'] = data['Category_1'].fillna('Unknown')
    data['Category_2'] = data['Category_2'].fillna('Unknown')
    
    return data

# Exploratory Data Analysis
def exploratory_analysis(data):
    print("\nPerforming exploratory data analysis...")
    
    # Basic statistics
    print("\nBasic statistics of the dataset:")
    print(data[['followers', 'Eng. (Auth.)', 'Eng. (Avg.)', 'engagement_rate']].describe())
    
    # Top 10 countries by number of influencers
    print("\nTop 10 countries by number of influencers:")
    country_counts = data['country'].value_counts().head(10)
    print(country_counts)
    
    # Top 10 categories
    print("\nTop 10 primary categories:")
    category_counts = data['Category_1'].value_counts().head(10)
    print(category_counts)
    
    # Correlation analysis
    print("\nCorrelation between followers and engagement:")
    correlation = data['followers'].corr(data['Eng. (Avg.)'])
    print(f"Correlation coefficient: {correlation:.4f}")
    
    return country_counts, category_counts

# Visualization functions
def create_visualizations(data, country_counts, category_counts):
    print("\nCreating visualizations...")
    
    # Create a directory for saving visualizations if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Followers distribution (log scale)
    plt.figure(figsize=(12, 6))
    sns.histplot(data['followers'], log_scale=True)
    plt.title('Distribution of Followers (Log Scale)')
    plt.xlabel('Number of Followers (Log Scale)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/followers_distribution.png')
    
    # 2. Engagement rate by category
    plt.figure(figsize=(14, 8))
    category_engagement = data.groupby('Category_1')['engagement_rate'].mean().sort_values(ascending=False).head(15)
    sns.barplot(x=category_engagement.index, y=category_engagement.values)
    plt.title('Average Engagement Rate by Category')
    plt.xlabel('Category')
    plt.ylabel('Average Engagement Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/engagement_by_category.png')
    
    # 3. Top 10 countries by influencers
    plt.figure(figsize=(12, 6))
    sns.barplot(x=country_counts.index, y=country_counts.values)
    plt.title('Top 10 Countries by Number of Influencers')
    plt.xlabel('Country')
    plt.ylabel('Number of Influencers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/top_countries.png')
    
    # 4. Followers vs. Engagement scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='followers', y='Eng. (Avg.)', hue='Category_1', data=data.sample(min(500, len(data))), alpha=0.7)
    plt.title('Followers vs. Average Engagement')
    plt.xlabel('Number of Followers')
    plt.ylabel('Average Engagement')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/followers_vs_engagement.png')
    
    # 5. Engagement rate by follower count ranges
    plt.figure(figsize=(12, 6))
    # Create follower count bins
    bins = [0, 1e6, 5e6, 10e6, 50e6, 100e6, float('inf')]
    labels = ['<1M', '1-5M', '5-10M', '10-50M', '50-100M', '>100M']
    data['follower_range'] = pd.cut(data['followers'], bins=bins, labels=labels)
    
    follower_engagement = data.groupby('follower_range')['engagement_rate'].mean()
    sns.barplot(x=follower_engagement.index, y=follower_engagement.values)
    plt.title('Average Engagement Rate by Follower Count Range')
    plt.xlabel('Follower Count Range')
    plt.ylabel('Average Engagement Rate (%)')
    plt.tight_layout()
    plt.savefig('visualizations/engagement_by_follower_range.png')
    
    print("Visualizations saved to 'visualizations' directory.")

# Advanced Analysis: Clustering influencers
def cluster_analysis(data):
    print("\nPerforming cluster analysis...")
    
    # Select features for clustering
    features = data[['followers', 'Eng. (Avg.)', 'engagement_rate']].copy()
    
    # Remove rows with missing values
    features = features.dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig('visualizations/elbow_method.png')
    
    # Choose k=4 clusters (this can be adjusted based on the elbow curve)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the original data
    clustered_data = features.copy()
    clustered_data['cluster'] = cluster_labels
    
    # Visualize the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='followers', y='engagement_rate', hue='cluster', data=clustered_data, palette='viridis')
    plt.title('Influencer Clusters based on Followers and Engagement Rate')
    plt.xlabel('Number of Followers')
    plt.ylabel('Engagement Rate (%)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('visualizations/influencer_clusters.png')
    
    # Analyze cluster characteristics
    print("\nCluster characteristics:")
    cluster_stats = clustered_data.groupby('cluster').mean()
    print(cluster_stats)
    
    return clustered_data, cluster_stats

# Predictive modeling: Predict engagement based on followers and other features
def predictive_modeling(data):
    print("\nBuilding predictive model for engagement...")
    
    # Prepare data for modeling
    model_data = data[['followers', 'engagement_rate', 'country']].copy()
    
    # Create dummy variables for top countries
    top_countries = data['country'].value_counts().head(10).index
    for country in top_countries:
        model_data[f'country_{country}'] = (model_data['country'] == country).astype(int)
    
    # Drop rows with missing values
    model_data = model_data.dropna()
    
    # Drop the original country column
    model_data = model_data.drop('country', axis=1)
    
    # Define features and target
    X = model_data.drop('engagement_rate', axis=1)
    y = model_data['engagement_rate']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature importance:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Predicting Engagement Rate')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    
    return model, feature_importance

# Main function to run the analysis
def main():
    # Load and preprocess data
    data = preprocess_data(df)
    
    # Perform exploratory analysis
    country_counts, category_counts = exploratory_analysis(data)
    
    # Create visualizations
    create_visualizations(data, country_counts, category_counts)
    
    # Perform cluster analysis
    clustered_data, cluster_stats = cluster_analysis(data)
    
    # Build predictive model
    model, feature_importance = predictive_modeling(data)
    
    print("\nAnalysis complete! Check the 'visualizations' directory for the generated plots.")
    
    # Generate insights report
    generate_insights_report(data, cluster_stats, feature_importance)

# Generate a text report with key insights
def generate_insights_report(data, cluster_stats, feature_importance):
    print("\nGenerating insights report...")
    
    with open('instagram_influencer_insights.txt', 'w') as f:
        f.write("INSTAGRAM INFLUENCER ANALYSIS INSIGHTS\n")
        f.write("=====================================\n\n")
        
        # Dataset overview
        f.write(f"Dataset Overview:\n")
        f.write(f"Total number of influencers: {len(data)}\n")
        f.write(f"Number of countries represented: {data['country'].nunique()}\n")
        f.write(f"Number of content categories: {data['Category_1'].nunique()}\n\n")
        
        # Follower statistics
        f.write(f"Follower Statistics:\n")
        f.write(f"Average followers: {data['followers'].mean():,.0f}\n")
        f.write(f"Median followers: {data['followers'].median():,.0f}\n")
        f.write(f"Maximum followers: {data['followers'].max():,.0f}\n\n")
        
        # Engagement statistics
        f.write(f"Engagement Statistics:\n")
        f.write(f"Average engagement rate: {data['engagement_rate'].mean():.2f}%\n")
        f.write(f"Median engagement rate: {data['engagement_rate'].median():.2f}%\n\n")
        
        # Top categories by engagement
        top_categories = data.groupby('Category_1')['engagement_rate'].mean().sort_values(ascending=False).head(5)
        f.write(f"Top 5 Categories by Engagement Rate:\n")
        for category, rate in top_categories.items():
            f.write(f"- {category}: {rate:.2f}%\n")
        f.write("\n")
        
        # Cluster insights
        f.write(f"Influencer Cluster Analysis:\n")
        for cluster, stats in cluster_stats.iterrows():
            f.write(f"Cluster {cluster}:\n")
            f.write(f"- Average followers: {stats['followers']:,.0f}\n")
            f.write(f"- Average engagement rate: {stats['engagement_rate']:.2f}%\n")
        f.write("\n")
        
        # Predictive factors
        f.write(f"Key Factors Influencing Engagement Rate:\n")
        for _, row in feature_importance.head(5).iterrows():
            f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        # Key insights and recommendations
        f.write(f"Key Insights and Recommendations:\n")
        f.write("1. The data shows an inverse relationship between follower count and engagement rate, ")
        f.write("suggesting that micro-influencers often have more engaged audiences.\n")
        f.write("2. Content categories with highest engagement rates should be prioritized for marketing campaigns.\n")
        f.write("3. Geographic targeting should consider both the number of influencers and their engagement rates.\n")
        f.write("4. Influencer selection should be based on the identified clusters to optimize campaign performance.\n")
    
    print("Insights report generated: 'instagram_influencer_insights.txt'")

# Run the analysis
if __name__ == "__main__":
    main()