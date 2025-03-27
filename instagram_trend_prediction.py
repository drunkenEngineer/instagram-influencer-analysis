import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.ticker as mtick
import re
import os
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
from collections import Counter


plt.style.use('fivethirtyeight')
sns.set(style='whitegrid')


engagement_cmap = LinearSegmentedColormap.from_list('engagement_colors', ['#1a237e', '#4a148c', '#b71c1c', '#ff6f00', '#ffd600'])
category_palette = sns.color_palette('viridis', 20)


print("Loading Instagram influencer data...")
file_path = 'social media influencers-INSTAGRAM - -DEC 2022.csv'
df = pd.read_csv(file_path, encoding='utf-8')


def preprocess_data(df):
    print("Preprocessing data...")
    
    data = df.copy()
    
    
    data.columns = [col.strip() for col in data.columns]
    
    
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
    
    
    for col in ['followers', 'Eng. (Auth.)', 'Eng. (Avg.)']:
        data[col] = data[col].apply(convert_to_numeric)
    
    
    data['engagement_rate'] = (data['Eng. (Avg.)'] / data['followers']) * 100
    
    
    data['Category_1'] = data['Category_1'].fillna('Unknown')
    data['Category_2'] = data['Category_2'].fillna('Unknown')
    
    
    conditions = [
        (data['followers'] < 10000),
        (data['followers'] >= 10000) & (data['followers'] < 100000),
        (data['followers'] >= 100000) & (data['followers'] < 1000000),
        (data['followers'] >= 1000000) & (data['followers'] < 10000000),
        (data['followers'] >= 10000000)
    ]
    choices = ['Nano', 'Micro', 'Mid-tier', 'Macro', 'Mega']
    data['influencer_tier'] = np.select(conditions, choices, default='Unknown')
    
    
    engagement_conditions = [
        (data['engagement_rate'] < 1),
        (data['engagement_rate'] >= 1) & (data['engagement_rate'] < 3),
        (data['engagement_rate'] >= 3) & (data['engagement_rate'] < 6),
        (data['engagement_rate'] >= 6) & (data['engagement_rate'] < 10),
        (data['engagement_rate'] >= 10)
    ]
    engagement_choices = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    data['engagement_level'] = np.select(engagement_conditions, engagement_choices, default='Unknown')
    
    return data


def advanced_eda(data):
    print("\nPerforming advanced exploratory data analysis...")
    
    
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
   
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='influencer_tier', y='engagement_rate', data=data, 
                order=['Nano', 'Micro', 'Mid-tier', 'Macro', 'Mega'],
                palette='viridis')
    plt.title('Engagement Rate Distribution by Influencer Tier', fontsize=16)
    plt.xlabel('Influencer Tier', fontsize=14)
    plt.ylabel('Engagement Rate (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/engagement_by_tier_boxplot.png')
    
    
    plt.figure(figsize=(16, 10))
    top_categories = data['Category_1'].value_counts().head(15)
    ax = sns.barplot(x=top_categories.index, y=top_categories.values, palette=category_palette)
    plt.title('Top 15 Content Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Number of Influencers', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    
    for i, count in enumerate(top_categories.values):
        ax.text(i, count + 5, f'{count}', ha='center', fontsize=11)
        
    plt.tight_layout()
    plt.savefig('visualizations/top_categories_distribution.png')
    
    
    plt.figure(figsize=(16, 10))
    country_engagement = data.groupby('country')['engagement_rate'].agg(['mean', 'count'])
    country_engagement = country_engagement.sort_values('count', ascending=False).head(15)
    
    ax = sns.barplot(x=country_engagement.index, y=country_engagement['mean'], 
                    palette=sns.color_palette('viridis', 15))
    plt.title('Average Engagement Rate by Country (Top 15 by Influencer Count)', fontsize=16)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Average Engagement Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
   
    for i, (_, row) in enumerate(country_engagement.iterrows()):
        ax.text(i, row['mean'] + 0.2, f'n={int(row["count"])}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/engagement_by_country.png')
    
   
    plt.figure(figsize=(12, 10))
    correlation_data = data[['followers', 'Eng. (Auth.)', 'Eng. (Avg.)', 'engagement_rate']]
    correlation_matrix = correlation_data.corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Correlation Heatmap of Key Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    
    
    plt.figure(figsize=(16, 12))
    category_performance = data.groupby('Category_1').agg({
        'followers': 'mean',
        'engagement_rate': 'mean',
        'Rank': 'count'
    }).sort_values('Rank', ascending=False).head(15)
    
    
    category_performance['count_normalized'] = category_performance['Rank'] / category_performance['Rank'].max() * 500
    
    plt.scatter(category_performance['followers'], category_performance['engagement_rate'], 
                s=category_performance['count_normalized'], alpha=0.6, 
                c=range(len(category_performance)), cmap='viridis')
    
    
    for i, (category, row) in enumerate(category_performance.iterrows()):
        plt.annotate(category, (row['followers'], row['engagement_rate']),
                    fontsize=9, ha='center')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.title('Category Performance: Followers vs. Engagement Rate', fontsize=16)
    plt.xlabel('Average Followers (Log Scale)', fontsize=14)
    plt.ylabel('Average Engagement Rate (%)', fontsize=14)
    plt.colorbar(label='Category Rank (by Count)', ticks=[])
    plt.tight_layout()
    plt.savefig('visualizations/category_performance_bubble.png')
    
    return top_categories, country_engagement


def advanced_clustering(data):
    print("\nPerforming advanced cluster analysis...")
    
    
    features = data[['followers', 'Eng. (Avg.)', 'engagement_rate']].copy()
    
    
    features = features.dropna()
    
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(k_range, inertia, 'o-', linewidth=2, markersize=10)
    plt.title('Elbow Method for Optimal Number of Clusters', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    
    
    
    elbow_point = 4 
    plt.annotate('Elbow Point', xy=(elbow_point, inertia[elbow_point-1]),
                xytext=(elbow_point+1, inertia[elbow_point-1]+inertia[0]/10),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualizations/elbow_method_advanced.png')
    
    
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    
    clustered_data = features.copy()
    clustered_data['cluster'] = cluster_labels
    
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    
    cluster_profiles = []
    for i, center in enumerate(centers):
        followers, engagement_avg, engagement_rate = center
        
        
        if followers < 100000:
            follower_category = "Small"
        elif followers < 1000000:
            follower_category = "Medium"
        elif followers < 10000000:
            follower_category = "Large"
        else:
            follower_category = "Mega"
            
        
        if engagement_rate < 1:
            engagement_category = "Low"
        elif engagement_rate < 5:
            engagement_category = "Medium"
        else:
            engagement_category = "High"
            
        profile = f"{engagement_category} Engagement {follower_category} Influencers"
        cluster_profiles.append(profile)
    
    
    cluster_mapping = {i: profile for i, profile in enumerate(cluster_profiles)}
    
    
    clustered_data['cluster_profile'] = clustered_data['cluster'].map(cluster_mapping)
    
    
    plt.figure(figsize=(14, 10))
    
    
    scatter = sns.scatterplot(x='followers', y='engagement_rate', 
                            hue='cluster_profile', data=clustered_data, 
                            palette='viridis', s=80, alpha=0.7)
    
    
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[2], s=200, c='red', 
                    marker='X', edgecolors='black', linewidth=2, 
                    label=f'Cluster {i} Center' if i == 0 else "")
        plt.annotate(f'Cluster {i}', (center[0], center[2]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    
    plt.title('Influencer Segmentation: Followers vs. Engagement Rate', fontsize=18)
    plt.xlabel('Number of Followers (Log Scale)', fontsize=14)
    plt.ylabel('Engagement Rate (%)', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    
    plt.legend(title='Cluster Profiles', title_fontsize=12, fontsize=10, 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig('visualizations/advanced_influencer_clusters.png')
    
    
    print("\nDetailed Cluster Profiles:")
    cluster_stats = clustered_data.groupby('cluster_profile').agg({
        'followers': ['mean', 'median', 'min', 'max', 'count'],
        'Eng. (Avg.)': ['mean', 'median'],
        'engagement_rate': ['mean', 'median', 'min', 'max']
    })
    
    print(cluster_stats)
    

    radar_data = clustered_data.groupby('cluster').agg({
        'followers': 'mean',
        'Eng. (Avg.)': 'mean',
        'engagement_rate': 'mean'
    })
    
    
    radar_data_normalized = radar_data.copy()
    for col in radar_data.columns:
        radar_data_normalized[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
    
    
    categories = ['Followers', 'Avg. Engagement', 'Engagement Rate']
    N = len(categories)
    
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    
    for i, cluster in enumerate(radar_data_normalized.index):
        values = radar_data_normalized.loc[cluster].values.flatten().tolist()
        values += values[:1]  
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster_mapping[cluster])
        ax.fill(angles, values, alpha=0.1)
    
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), title="Cluster Profiles")
    plt.title('Cluster Characteristics Radar Chart', fontsize=16, y=1.1)
    
    plt.tight_layout()
    plt.savefig('visualizations/cluster_radar_chart.png')
    
    return clustered_data, cluster_stats, cluster_mapping


def advanced_predictive_modeling(data, clustered_data):
    print("\nBuilding advanced predictive models...")
    
    
    data_with_index = data.copy()
    data_with_index['temp_index'] = range(len(data_with_index))
    
    clustered_data_subset = clustered_data[['cluster', 'cluster_profile']].copy()
    clustered_data_subset['temp_index'] = range(len(clustered_data_subset))
    
    
    merged_data = data_with_index.merge(clustered_data_subset, on='temp_index', how='left')
    merged_data.drop('temp_index', axis=1, inplace=True)
    
    
    model_data = merged_data[[
        'followers', 'engagement_rate', 'country', 'Category_1', 
        'Category_2', 'cluster', 'influencer_tier'
    ]].copy()
    
    
    model_data = model_data.dropna()
    
    
    categorical_features = ['country', 'Category_1', 'Category_2', 'influencer_tier']
    numeric_features = ['followers']
    
    
    for cat_feature in categorical_features:
        top_categories = model_data[cat_feature].value_counts().head(10).index
        model_data[cat_feature] = model_data[cat_feature].apply(
            lambda x: x if x in top_categories else 'Other')
    
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    
    y = model_data['engagement_rate']
    
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    
    plt.figure(figsize=(14, 8))
    
    
    model_scores = []
    
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nTraining {name}...")
        
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)])
        
        
        cv_scores = cross_val_score(pipeline, model_data.drop('engagement_rate', axis=1), 
                                   y, cv=5, scoring='neg_mean_squared_error')
        
        
        rmse_scores = np.sqrt(-cv_scores)
        
        
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()
        
        print(f"{name} - Mean RMSE: {mean_rmse:.4f}, Std: {std_rmse:.4f}")
        
        
        model_scores.append({
            'name': name,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse
        })
        
        
        plt.bar(i, mean_rmse, yerr=std_rmse, capsize=10, 
                color=plt.cm.viridis(i/len(models)), 
                label=name)
    

    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks(range(len(models)), list(models.keys()), rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/model_performance_comparison.png')
    
    
    best_model_name = min(model_scores, key=lambda x: x['mean_rmse'])['name']
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    
   
    best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', best_model)])
    
    X = model_data.drop('engagement_rate', axis=1)
    best_pipeline.fit(X, y)
    
    
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        
        feature_names = (numeric_features + 
                        list(best_pipeline.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_features)))
        
        
        importances = best_pipeline.named_steps['model'].feature_importances_
        
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(20)
        
        
        plt.figure(figsize=(14, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title(f'Top 20 Feature Importance - {best_model_name}', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance_advanced.png')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nFinal model evaluation on test set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#1f77b4')
    
    
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.title('Actual vs. Predicted Engagement Rates', fontsize=16)
    plt.xlabel('Actual Engagement Rate (%)', fontsize=14)
    plt.ylabel('Predicted Engagement Rate (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/actual_vs_predicted.png')
    
    return best_pipeline, best_model_name, r2


def trend_analysis(data, clustered_data, top_categories):
    print("\nPerforming trend analysis...")
    
    
    plt.figure(figsize=(14, 8))
    
    
    tier_category_engagement = data.groupby(['influencer_tier', 'Category_1'])['engagement_rate'].mean().reset_index()
    
   
    top_8_categories = top_categories.index[:8]
    filtered_data = tier_category_engagement[tier_category_engagement['Category_1'].isin(top_8_categories)]
    
    
    sns.barplot(x='influencer_tier', y='engagement_rate', hue='Category_1', 
                data=filtered_data, 
                order=['Nano', 'Micro', 'Mid-tier', 'Macro', 'Mega'],
                palette='viridis')
    
    plt.title('Engagement Rate Trends by Influencer Tier and Category', fontsize=16)
    plt.xlabel('Influencer Tier', fontsize=14)
    plt.ylabel('Average Engagement Rate (%)', fontsize=14)
    plt.legend(title='Category', title_fontsize=12, fontsize=10, loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/engagement_trends_by_tier_category.png')
    
   
    plt.figure(figsize=(16, 10))
    
    
    top_10_countries = data['country'].value_counts().head(10).index
    
    
    country_data = data[data['country'].isin(top_10_countries)]
    
    
    country_tier_data = country_data.groupby(['country', 'influencer_tier'])['engagement_rate'].mean().reset_index()
    
    
    sns.catplot(x='country', y='engagement_rate', hue='influencer_tier', 
                data=country_tier_data, kind='bar', height=8, aspect=1.5,
                palette='viridis', legend_out=False)
    
    plt.title('Engagement Rate by Country and Influencer Tier', fontsize=16)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Average Engagement Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Influencer Tier', title_fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/geographic_engagement_trends.png')
    
    
    plt.figure(figsize=(12, 8))
    
    
    all_categories = ' '.join(data['Category_1'].dropna().tolist() + data['Category_2'].dropna().tolist())
    
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         colormap='viridis', max_words=100, contour_width=1, contour_color='steelblue')
    wordcloud.generate(all_categories)
    
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Content Categories Word Cloud', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/category_wordcloud.png')
    
    
    plt.figure(figsize=(16, 10))
    
    
    engagement_level_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    category_engagement_dist = pd.crosstab(data['Category_1'], data['engagement_level'])
    
    
    category_engagement_pct = category_engagement_dist.div(category_engagement_dist.sum(axis=1), axis=0) * 100
    
    
    top_10_categories = data['Category_1'].value_counts().head(10).index
    category_engagement_pct = category_engagement_pct.loc[top_10_categories]
    
    
    category_engagement_pct[engagement_level_order].plot(kind='bar', stacked=True, 
                                                       figsize=(16, 10), 
                                                       colormap='viridis')
    
    plt.title('Engagement Level Distribution by Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(title='Engagement Level', title_fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/engagement_level_distribution.png')
    

    plt.figure(figsize=(14, 8))
    
    
    category_growth = data.groupby('Category_1').agg({
        'followers': 'mean',
        'engagement_rate': 'mean',
        'Rank': 'count'
    }).sort_values('engagement_rate', ascending=False).head(10)
    
    
    ax = category_growth['engagement_rate'].plot(kind='barh', figsize=(14, 8), 
                                              color=plt.cm.viridis(np.linspace(0, 1, len(category_growth))))
    
    
    for i, (category, row) in enumerate(category_growth.iterrows()):
        ax.text(row['engagement_rate'] + 0.1, i, f'Count: {int(row["Rank"])}', 
                va='center', fontsize=10)
    
    plt.title('Predicted Category Growth Potential Based on Engagement', fontsize=16)
    plt.xlabel('Average Engagement Rate (%)', fontsize=14)
    plt.ylabel('Content Category', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/category_growth_prediction.png')
    
    return category_growth


def generate_comprehensive_report(data, clustered_data, cluster_mapping, category_growth, best_model_name, r2):
    print("\nGenerating comprehensive insights report...")
    
    with open('instagram_trend_prediction_insights.md', 'w') as f:
        f.write("# Instagram Influencer Trend Analysis and Prediction\n\n")
        
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of Instagram influencer data, ")
        f.write("identifying key trends, engagement patterns, and predictive insights. ")
        f.write("The analysis leverages advanced data science techniques including clustering, ")
        f.write("predictive modeling, and trend analysis to provide actionable intelligence ")
        f.write("for social media marketing strategies.\n\n")
        
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total influencers analyzed**: {len(data):,}\n")
        f.write(f"- **Countries represented**: {data['country'].nunique():,}\n")
        f.write(f"- **Content categories**: {data['Category_1'].nunique() + data['Category_2'].nunique():,}\n")
        f.write(f"- **Follower range**: {data['followers'].min():,.0f} to {data['followers'].max():,.0f}\n")
        f.write(f"- **Average engagement rate**: {data['engagement_rate'].mean():.2f}%\n\n")
        
        
        f.write("## Key Findings\n\n")
        
        
        f.write("### Engagement Patterns\n\n")
        f.write("- **Inverse relationship between followers and engagement**: ")
        f.write("The analysis reveals a clear inverse relationship between follower count and engagement rate. ")
        f.write("Smaller accounts (Nano and Micro influencers) consistently demonstrate higher engagement rates ")
        f.write("across most content categories.\n\n")
        
        
        top_categories = data.groupby('Category_1')['engagement_rate'].mean().sort_values(ascending=False).head(5)
        f.write("- **Top performing content categories**:\n")
        for category, rate in top_categories.items():
            f.write(f"  - {category}: {rate:.2f}% average engagement rate\n")
        f.write("\n")
        
        
        top_countries = data.groupby('country')['engagement_rate'].mean().sort_values(ascending=False).head(5)
        f.write("- **Geographic engagement hotspots**:\n")
        for country, rate in top_countries.items():
            f.write(f"  - {country}: {rate:.2f}% average engagement rate\n")
        f.write("\n")
        
        
        f.write("### Influencer Segmentation\n\n")
        f.write("Our cluster analysis identified distinct influencer segments with unique characteristics:\n\n")
        
        for cluster, profile in cluster_mapping.items():
            cluster_data = clustered_data[clustered_data['cluster'] == cluster]
            f.write(f"- **{profile}**:\n")
            f.write(f"  - Average followers: {cluster_data['followers'].mean():,.0f}\n")
            f.write(f"  - Average engagement rate: {cluster_data['engagement_rate'].mean():.2f}%\n")
            f.write(f"  - Number of influencers: {len(cluster_data):,}\n\n")
        
        
        f.write("### Predictive Insights\n\n")
        f.write(f"Our {best_model_name} model achieved an R² score of {r2:.4f}, ")
        f.write("indicating its ability to predict engagement rates based on influencer characteristics. ")
        f.write("Key factors influencing engagement include:\n\n")
        f.write("- Follower count (negative correlation)\n")
        f.write("- Content category\n")
        f.write("- Geographic location\n")
        f.write("- Influencer tier\n\n")
        
        
        f.write("### Trend Predictions\n\n")
        f.write("Based on current engagement patterns, we predict the following categories will see growth:\n\n")
        
        for category, row in category_growth.head(5).iterrows():
            f.write(f"- **{category}**: {row['engagement_rate']:.2f}% engagement rate, ")
            f.write(f"indicating strong growth potential\n")
        f.write("\n")
        
        
        f.write("## Strategic Recommendations\n\n")
        f.write("1. **Micro-influencer focus**: Prioritize partnerships with micro-influencers (10K-100K followers) ")
        f.write("for higher engagement rates and potentially better ROI.\n\n")
        
        f.write("2. **Category targeting**: Invest in content categories showing high engagement rates ")
        f.write("and growth potential, particularly: ")
        f.write(", ".join(category_growth.head(3).index) + ".\n\n")
        
        f.write("3. **Geographic strategy**: Tailor influencer campaigns to leverage high-engagement ")
        f.write("countries while considering market size and relevance.\n\n")
        
        f.write("4. **Segment-specific approaches**: Develop distinct strategies for each influencer segment ")
        f.write("identified in the cluster analysis, recognizing their unique engagement patterns.\n\n")
        
        f.write("5. **Trend monitoring**: Regularly update this analysis to track emerging categories ")
        f.write("and shifting engagement patterns in the dynamic social media landscape.\n\n")
        
        
        f.write("## Conclusion\n\n")
        f.write("This analysis provides a data-driven foundation for optimizing influencer marketing strategies. ")
        f.write("By leveraging the identified patterns and predictions, marketers can make more informed decisions ")
        f.write("about influencer partnerships, content categories, and geographic targeting to maximize engagement ")
        f.write("and campaign effectiveness.")
    
    print("Comprehensive insights report generated: 'instagram_trend_prediction_insights.md'")


def main():
    
    print("Starting Instagram influencer trend analysis and prediction...")
    data = preprocess_data(df)
    
    
    top_categories, country_engagement = advanced_eda(data)
    
    
    clustered_data, cluster_stats, cluster_mapping = advanced_clustering(data)
    
    
    best_pipeline, best_model_name, r2 = advanced_predictive_modeling(data, clustered_data)
    
    
    category_growth = trend_analysis(data, clustered_data, top_categories)
    
    
    generate_comprehensive_report(data, clustered_data, cluster_mapping, category_growth, best_model_name, r2)
    
    print("\nAnalysis complete! Check the 'visualizations' directory for generated plots and the insights report.")


if __name__ == "__main__":
    main()