# File: data_preprocessing.py

import os
import pandas as pd
from langdetect import detect
import logging
import re
from fuzzywuzzy import process
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(filename = 'data_preprocessing.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load job postings from CSV
input_filename = 'job_postings\data_scientist_jobs.csv'
output_filename = 'job_postings\preprocessed_data_scientist_jobs.csv'

def is_english(text):
    """
    Check if the given text is in English.

    Args:
        text (str): Text to check.

    Returns:
        bool: True if text is in English, False otherwise.
    """
    try:
        return detect(text) == 'en'
    except:
        return False

def normalize_job_title(title):
    """
    Normalize job titles and create a seniority level feature.

    Args:
        title (str): Job title.

    Returns:
        tuple: Normalized job title and seniority level.
    """
    title = title.lower()
    
    if any(term in title for term in ['junior', 'jr', 'intern']):
        seniority = 'junior'
    elif any(term in title for term in ['senior', 'sr', 'manager', 'lead', 'principal', 'director', 'head']):
        seniority = 'senior'
    elif any(term in title for term in ['staff', 'mid', 'middle']):
        seniority = 'mid'
    else:
        seniority = 'unknown'
    
    return title, seniority

def group_locations(location):
    """
    Group locations into broader areas using regex and fuzzy matching.

    Args:
        location (str): Location name.

    Returns:
        str: Broader location category.
    """
    location = location.lower().strip()
    patterns = {
        'New York Metro': r'new york|nyc|manhattan|brooklyn|queens|bronx',
        'Los Angeles': r'los angeles|la|l.a.',
        'San Francisco': r'san francisco|sf|s.f.',
        'Seattle, WA': r'seattle',
        'Remote': r'remote|anywhere',
        'Boston, MA': r'boston',
        'Chicago, IL': r'chicago',
        'Denver, CO': r'denver',
        'Washington, DC': r'washington|dc',
        'New Jersey': r'new jersey|nj|jersey city|newark',
        'Pennsylvania': r'pennsylvania|pa|philadelphia|pittsburgh',
        'Texas': r'texas|tx|houston|dallas|austin|san antonio',
        'Florida': r'florida|fl|miami|orlando|tampa',
        'Alabama': r'alabama|al',
        'Alaska': r'alaska|ak',
        'Arizona': r'arizona|az',
        'Arkansas': r'arkansas|ar',
        'California': r'california|ca|los angeles|san francisco|san diego',
        'Colorado': r'colorado|co|denver',
        'Connecticut': r'connecticut|ct',
        'Delaware': r'delaware|de',
        'Georgia': r'georgia|ga|atlanta',
        'Hawaii': r'hawaii|hi',
        'Idaho': r'idaho|id',
        'Illinois': r'illinois|il|chicago',
        'Indiana': r'indiana|in',
        'Iowa': r'iowa|ia',
        'Kansas': r'kansas|ks',
        'Kentucky': r'kentucky|ky',
        'Louisiana': r'louisiana|la',
        'Maine': r'maine|me',
        'Maryland': r'maryland|md',
        'Massachusetts': r'massachusetts|ma|boston',
        'Michigan': r'michigan|mi',
        'Minnesota': r'minnesota|mn',
        'Mississippi': r'mississippi|ms',
        'Missouri': r'missouri|mo',
        'Montana': r'montana|mt',
        'Nebraska': r'nebraska|ne',
        'Nevada': r'nevada|nv',
        'New Hampshire': r'new hampshire|nh',
        'New Mexico': r'new mexico|nm',
        'North Carolina': r'north carolina|nc',
        'North Dakota': r'north dakota|nd',
        'Ohio': r'ohio|oh',
        'Oklahoma': r'oklahoma|ok',
        'Oregon': r'oregon|or',
        'Pennsylvania': r'pennsylvania|pa|philadelphia|pittsburgh',
        'Rhode Island': r'rhode island|ri',
        'South Carolina': r'south carolina|sc',
        'South Dakota': r'south dakota|sd',
        'Tennessee': r'tennessee|tn',
        'Utah': r'utah|ut',
        'Vermont': r'vermont|vt',
        'Virginia': r'virginia|va',
        'Washington': r'washington|wa|seattle',
        'West Virginia': r'west virginia|wv',
        'Wisconsin': r'wisconsin|wi',
        'Wyoming': r'wyoming|wy'
    }
    
    for key, pattern in patterns.items():
        if re.search(pattern, location):
            return key
    # Fuzzy match for additional robustness
    location_choices = list(patterns.keys())
    match, score = process.extractOne(location, location_choices)
    if score > 80:  # Threshold for fuzzy matching
        return match
    return 'Other'

def extract_skills(description):
    """
    Extract skills from job descriptions.

    Args:
        description (str): Job description.

    Returns:
        list: List of skills mentioned in the description.
    """
    # Expanded list of skills
    skills = [
        'python', ' r ', 'sql', 'java', 'javascript', 'c++', 'c#', 'machine learning',
        'data analysis', 'data visualization', 'deep learning', 'neural networks',
        'nlp', 'natural language processing', 'data mining', 'big data', 'hadoop',
        'spark', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'matplotlib',
        'seaborn', 'ggplot2', 'tableau', 'power bi', 'aws', 'azure', 'gcp', 'cloud computing',
        'sas', 'stata', 'data warehousing', 'etl', 'hive', 'pig', 'mapreduce', 'scala',
        'matlab', 'excel', 'nosql', 'mongodb', 'postgresql', 'git', 'github', 'flask',
        'django', 'shell scripting', 'bash', 'linux', 'unix', 'docker', 'kubernetes',
        'ansible', 'jenkins', 'agile', 'scrum', 'project management', 'statistics',
        'probability', 'regression', 'classification', 'clustering', 'time series',
        'forecasting', 'data engineering', 'feature engineering', 'model deployment',
        'a/b testing', 'experimentation', 'data governance', 'data quality', 'data ethics',
        'calculus', 'linear algebra', 'causal analysis', 'hypothesis testing', 'bayesian statistics',
        'data wrangling', 'data cleaning', 'data preprocessing', 'statistical modeling', 'random forests',
        'support vector machines', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost'
    ]
    found_skills = [skill for skill in skills if skill in description]
    return found_skills

def convert_relative_time(relative_time):
    current_time = datetime.now()
    
    match = re.match(r'(\d+)\s*(\w+)\s*ago', relative_time)
    
    if not match:
        return current_time

    value, unit = int(match.group(1)), match.group(2)
    if 'day' in unit:
        return (current_time - timedelta(days=value)).date()
    elif 'week' in unit:
        return (current_time - timedelta(weeks=value)).date()
    elif 'month' in unit:
        return (current_time - timedelta(days=value*30)).date()
    elif 'hour' in unit:
        return (current_time - timedelta(hours=value)).date()
    else:
        return current_time.date()

def preprocess_data(input_file, output_file):
    """
    Preprocess the job postings data.

    Args:
        input_file (str): Input CSV file with raw job postings.
        output_file (str): Output CSV file to save preprocessed data.
    """
    logging.info("Starting data preprocessing...")
    data = pd.read_csv(input_file)

    # Filter out non-English descriptions
    data = data[data['Description'].apply(is_english)]

    # Normalize job titles and create seniority levels
    data['JobTitle'], data['SeniorityLevel'] = zip(*data['Title'].apply(normalize_job_title))

    # Convert relevant columns to lowercase
    data['Location'] = data['Location'].str.lower()
    data['Description'] = data['Description'].str.lower()
    data['Highlights'] = data['Highlights'].str.lower()

    # Group and clean locations
    data['GeneralLocation'] = data['Location'].apply(group_locations)

    # Extract skills from job descriptions
    data['Skills'] = data['Description'].apply(extract_skills)

    # Fill missing dates with current day
    data['Date'] = data['Date'].fillna('0 days ago')
    data['Date'] = data['Date'].apply(convert_relative_time)

    # Select and reorder columns
    preprocessed_data = data[['JobID', 'JobTitle', 'SeniorityLevel', 'Company', 'GeneralLocation', 'Description', 'Highlights', 'Skills', 'Date']]


    # Save preprocessed data to CSV
    if not os.path.isfile(output_file):
        preprocessed_data.to_csv(output_file, index=False)
    else: # else it exists so append without writing the header
        preprocessed_data.to_csv(output_file, mode='a', header=False, index=False)

    logging.info(f"Preprocessed data saved to {output_file}")

if __name__ == '__main__':
    preprocess_data(input_filename, output_filename)
