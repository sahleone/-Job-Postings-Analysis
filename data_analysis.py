
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from wordcloud import WordCloud
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
import warnings

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Set up logging
logging.basicConfig(filename = 'data_analysis.log ',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Filter out specific warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_data(input_file):
    """
    Load preprocessed job postings data.

    Args:
        input_file (str): Path to the preprocessed CSV file.

    Returns:
        DataFrame: Loaded job postings data.
    """
    logging.info(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, encoding='unicode_escape')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Skills'] = data['Skills'].apply(eval)

    return data

def analyze_daily_postings(data):
    """
    Analyze the daily number of job postings.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing daily number of job postings...")
    daily_postings = data.groupby(data['Date'].dt.date).size()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=daily_postings.index, y=daily_postings.values)
    plt.title('Daily Number of Job Postings')
    plt.xlabel('Date')
    plt.ylabel('Number of Postings')
    plt.grid(True)
    plt.savefig(os.path.join(fig_save_path ,'daily_postings.png'))
    plt.tight_layout()
    plt.close()
    logging.info("Daily number of job postings analysis completed. Saved as daily_postings.png.")

def analyze_postings_by_location(data):
    """
    Analyze the distribution of job postings by location.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing job postings by location...")
    location_postings = data['GeneralLocation'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=location_postings.index, y=location_postings.values)
    plt.title('Job Postings by General Location')
    plt.xlabel('Location')
    plt.ylabel('Number of Postings')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.savefig(os.path.join(fig_save_path ,'postings_by_location.png'))
    plt.tight_layout()
    plt.close()
    logging.info("Job postings by location analysis completed. Saved as postings_by_location.png.")

def analyze_skills(data):
    """
    Analyze and visualize the most demanded skills.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing skills from job descriptions...")
    skills_series = data['Skills'].explode()
    top_skills = skills_series.value_counts().head(20)  # Top 20 skills

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_skills.index, y=top_skills.values)
    plt.title('Top 20 Skills in Job Descriptions')
    plt.xlabel('Skills')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path ,'top_skills.png'))
    plt.close()

    logging.info("Skills analysis completed. Saved as top_skills.png.")

def analyze_cooccurring_skills(data):
    """
    Perform Apriori analysis to find frequent itemsets and association rules among skills.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Performing Apriori analysis on skills...")
    # Create a one-hot encoded dataframe for skills
    all_skills = set(skill for sublist in data['Skills'] for skill in sublist)
    encoded_data = pd.DataFrame([{skill: (skill in skills) for skill in all_skills} for skills in data['Skills']])

    # Perform Apriori analysis
    frequent_itemsets = apriori(encoded_data, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    # Save results
    frequent_itemsets.to_csv(os.path.join(data_save_path,'frequent_itemsets.csv'), index=False)
    rules.to_csv(os.path.join(data_save_path,'association_rules.csv'), index=False)

    logging.info("Apriori analysis completed. Results saved to frequent_itemsets.csv and association_rules.csv.")

    # Visualize the top 10 frequent itemsets
    top_itemsets = frequent_itemsets.nlargest(20, 'support')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='support', y=top_itemsets['itemsets'].astype(str), data=top_itemsets)
    plt.title('Top 20 Frequent Itemsets')
    plt.xlabel('Support')
    plt.ylabel('Itemsets')
    plt.savefig(os.path.join(fig_save_path, 'top_frequent_itemsets.png'))
    plt.close()
    logging.info("Saved as top_frequent_itemsets.png")

    # Visualize the association rules using a network graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for _, rule in tqdm(rules.iterrows()):
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_node(antecedent, label=antecedent)
                G.add_node(consequent, label=consequent)
                G.add_edge(antecedent, consequent, weight=rule['lift'], label=f"conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f}")

    # Define position for each node
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')

    # Draw edges
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, edge_color='grey')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.savefig(os.path.join(fig_save_path, 'Network Graph of Association Rules.png'))
    plt.close()

    logging.info("Visualization of Apriori analysis completed. Saved as top_frequent_itemsets.png and association_rules_graph.png.")


def analyze_remote_work(data):
    """
    Analyze the proportion of remote vs. on-site job postings.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing remote vs. on-site job postings...")
    if 'Highlights' in data.columns:
        data['Remote'] = data['Highlights'].apply(lambda x: 'remote' in x if pd.notnull(x) else False)
        remote_work_counts = data['Remote'].value_counts()

        plt.figure(figsize=(10, 6))
        remote_work_counts.plot(kind='pie', autopct='%1.1f%%', labels=['On-site', 'Remote'],\
                                colors=['#000000','#C8102E'])
        plt.title('Remote vs. On-site Job Postings')
        plt.ylabel('')
        plt.savefig(os.path.join(fig_save_path ,'remote_vs_onsite.png'))
        plt.tight_layout()
        plt.close()
        logging.info("Remote work analysis completed. Saved as remote_vs_onsite.png.")
    else:
        logging.warning("Highlights column not found in the data.")

def analyze_company_postings(data):
    """
    Analyze job postings at the company level to identify which companies are posting the most jobs and what skills they demand.

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing job postings by company...")

    # Top companies by number of postings
    top_companies = data['Company'].value_counts().head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_companies.index, y=top_companies.values)
    plt.title('Top 20 Companies by Job Postings')
    plt.xlabel('Company')
    plt.ylabel('Number of Postings')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path ,'top_companies.png'))
    logging.info("Top companies by job postings analysis completed. Saved as top_companies.png.")

    # Skills demanded by top companies
    company_skills = data.groupby('Company')['Skills'].apply(lambda x: [skill for sublist in x for skill in sublist])
    top_company_skills = company_skills.loc[top_companies.index]

    for company, skills in top_company_skills.items():
        skill_counts = pd.Series(skills).value_counts().head(10)  # Top 10 skills for each company
        plt.figure(figsize=(10, 6))
        sns.barplot(x=skill_counts.index, y=skill_counts.values)
        plt.title(f'Top Skills in {company}')
        plt.xlabel('Skills')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.savefig(os.path.join(fig_save_path ,f'top_skills_{company}.png'))
        plt.close()
        logging.info(f"Top skills for {company} analysis completed. Saved as top_skills_{company}.png.")

def clean_text(text):
    """
    Clean the input text by removing stopwords and any odd characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Define stopwords
    stop_words = set(stopwords.words('english')).union(["job", "experience", "work", "skills","data","science"])

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords and non-alphabetic characters
    cleaned_words = [word for word in words if word.isalpha() and word.lower() not in stop_words]

    return ' '.join(cleaned_words)


def create_word_cloud(text, title, output_file, weights=None, stopwords = None):  # Added weights argument
    # Customized WordCloud 
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white', 
        stopwords = stopwords,
        min_font_size=10,  # Adjust minimum font size for readability
        max_words=150  # Limit the number of words displayed
    )
    if weights:
        wordcloud.generate_from_frequencies(weights)
    else:
        wordcloud.generate(text)
    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(fig_save_path, output_file))
    plt.close()
    logging.info(f"{title} word cloud saved as {output_file}")

def analyze_word_clouds(data):
    logging.info("Creating word clouds for job descriptions and highlights...")
    

    for column in ['Description', 'Highlights']:
        text = ' '.join(data[column].dropna())

        # Enhanced Cleaning (Lowercase, remove numbers, special characters)
        cleaned_text = clean_text(text)
        custom_stopwords = set(stopwords.words('english')).union(["job", "experience", "work", "skills", "company", "benefits", "business","position","data","science"])
        custom_stopwords = list(custom_stopwords.union(data["Company"]))

        # TF-IDF Calculation
        vectorizer = TfidfVectorizer(stop_words=custom_stopwords, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(data[column].dropna())
        feature_names = vectorizer.get_feature_names_out()  # Get feature names
        tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Sum TF-IDF scores across documents

        # Create dictionary of words and their TF-IDF scores
        tfidf_dict = dict(zip(feature_names, tfidf_scores))
        
        # Plot and Save with TF-IDF
        title = f"{column} Word Cloud (TF-IDF)"  # More dynamic title
        output_file = f"{column.lower()}_tfidf_word_cloud.png"
        create_word_cloud(cleaned_text, title, output_file, weights=tfidf_dict, stopwords = custom_stopwords) 


def analyze_correlations(data, top_n_skills=15):
    logging.info("Performing correlation analysis...")

    # Handle missing data for relevant columns
    data_filled = data.copy().fillna({'SeniorityLevel': 'Unknown', 'Remote': 'Unknown'})

    # One-hot encode categorical features
    categorical_features = ['SeniorityLevel', 'Remote']
    data_encoded = pd.get_dummies(data_filled, columns=categorical_features)

    # Expand skills and one-hot encode them
    all_skills = set(skill for sublist in data['Skills'] for skill in sublist)
    skill_dummies = data['Skills'].apply(lambda x: pd.Series({skill: 1 if skill in x else 0 for skill in all_skills}))
    data_encoded = pd.concat([data_encoded, skill_dummies], axis=1).fillna(0)

    # Filter out non-numeric columns
    numeric_data = data_encoded.select_dtypes(include=[np.number])

    # Calculate correlation matrix for all data
    correlation_matrix = numeric_data.corr()

    # Plot correlation matrix for all data (using all skills)
    _plot_correlation_matrix(correlation_matrix, skill_dummies.columns, "All Seniority Levels")

    # Plot for each seniority level
    for seniority in ['junior','senior','mid']:
        seniority_mask = data_filled['SeniorityLevel'] == seniority
        seniority_data = numeric_data.loc[seniority_mask]
        
        # Calculate correlation matrix for this seniority level
        corr_matrix_seniority = seniority_data.corr()

        # Select top skills for this seniority level
        avg_abs_corr = corr_matrix_seniority[skill_dummies.columns].abs().mean()
        top_skills_seniority = avg_abs_corr.sort_values(ascending=False)[:top_n_skills]  
        
        # Save top skills for this seniority to CSV
        top_skills_seniority.to_csv(os.path.join(data_save_path, f'top_skills_{seniority}.csv'))
        
        # Plot correlation matrix for this seniority level (using top skills for this seniority)
        _plot_correlation_matrix(corr_matrix_seniority, top_skills_seniority.index, seniority)



def _plot_correlation_matrix(correlation_matrix, top_skills, title):
    skill_mask = correlation_matrix.columns.isin(top_skills)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        correlation_matrix.loc[skill_mask, skill_mask], 
        annot=True, 
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
    )
    plt.title(f"Correlation Matrix (Top {len(top_skills)} Skills) - {title}")
    plt.savefig(os.path.join(fig_save_path, f"correlation_matrix_{title.replace(' ', '_')}.png"))
    plt.tight_layout()
    plt.close()

def analyze_role_SeniorityLevel(data):
    """
    Analyze the SeniorityLevel of roles

    Args:
        data (DataFrame): Job postings data.
    """
    logging.info("Analyzing the SeniorityLevel of roles...")
    if 'Highlights' in data.columns:
        counts = data['SeniorityLevel'].value_counts()

        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title('SeniorityLevel count')
        plt.ylabel('')
        plt.savefig(os.path.join(fig_save_path ,'SeniorityLevel count.png'))
        plt.tight_layout()
        plt.close()
        logging.info("SeniorityLevel of roles analysis completed. Saved as SeniorityLevel count.png.")
    else:
        logging.warning("Highlights column not found in the data.")

def main():
    """
    Main function to perform data analysis on job postings.
    """
    path = 'charts'
    if not os.path.exists(path):
        os.makedirs(path)
       
    data = load_data(input_filename)
    analyze_daily_postings(data)
    analyze_postings_by_location(data)
    analyze_skills(data)
    # analyze_cooccurring_skills(data)
    analyze_remote_work(data)
    analyze_role_SeniorityLevel(data)
    # analyze_company_postings(data)
    analyze_word_clouds(data)

    # split by company / location/ industry ?
    analyze_correlations(data)

if __name__ == '__main__':

    # Load preprocessed job postings data
    input_filename = 'job_postings\\New York\preprocessed_data_scientist_jobs.csv'

    # Ensure the 'chart' directory exists
    fig_save_path = 'charts'
    if not os.path.exists(fig_save_path ):
        os.makedirs(fig_save_path )

    # data path
    data_save_path = 'job_postings/'
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    main()
