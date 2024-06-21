
# File: data_collection.py

import os
import requests
import json
import csv
from dotenv import load_dotenv
import logging
from serpapi import GoogleSearch

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(filename='data_collection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the API key and endpoint
API_KEY = os.getenv('SERPER_API_KEY')

# Define search parameters
params = {
    'engine': 'google_jobs',
    'q': 'Data Scientist',
    'api_key': API_KEY,
    'location': 'New York',
    'hl': 'en',  # Set language to English
    'start': 0,   # Pagination start parameter
    'chips':'date_posted:month'
}

# Ensure the folder exists
output_folder = 'job_postings'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fetch_job_listings(params):
    """
    Fetch job listings from the SerpApi.

    Args:
        params (dict): Parameters for the API request.

    Returns:
        tuple: A tuple containing a list of job listings and the full response data.
    """
    try:
        logging.info(f"Fetching job listings starting from {params['start']}...")
        search = GoogleSearch(params)
        data = search.get_dict()
        return data.get('jobs_results', []), data
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return [], {}
    


def save_response_as_json(response, start):
    """
    Save API response as a JSON file.

    Args:
        response (dict): The full response data from the API.
        start (int): The start parameter used for the API request.
    """
    filename = os.path.join(output_folder,params['location'], f'job_postings_{start}.json')
    try:
        logging.info(f"Saving response to {filename}...")
        with open(filename, 'w') as file:
            json.dump(response, file, indent=4)
        logging.info(f"Response saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Failed to save response: {e}")


def merge_highlights(highlights):
    """
    Merge job highlights into a single string of text.

    Args:
        highlights (list): List of job highlights.

    Returns:
        str: Merged highlights as a single string.
    """
    if highlights:
        return ' '.join([highlight.get('title', '') + ': ' + ' '.join(highlight.get('items', [])) for highlight in highlights])
    return ''


def save_to_csv(job_listings, filename='data_scientist_jobs.csv'):
    """
    Save job listings to a CSV file.

    Args:
        job_listings (list): List of job listings.
        filename (str): Name of the CSV file.
    """
    filename='data_scientist_jobs.csv'

    file_path = os.path.join(output_folder,params['location'],filename)

    file_exists = os.path.isfile(file_path)

    try:
        logging.info(f"Saving {len(job_listings)} job postings to {filename}...")
        with open(file_path, mode='a', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            # writes the header if file does not exist
            if not file_exists:
                writer.writerow(['JobID', 'Title', 'Company', 'Location', 'Description', 'Highlights', 'Date'])
            for job in job_listings:
                job_highlights = merge_highlights(job.get('job_highlights', []))
                writer.writerow([job.get('job_id'), job.get('title'), job.get('company_name'), job.get('location'), job.get('description'), job_highlights, job.get('detected_extensions', {}).get('posted_at')])
        logging.info(f"Data saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")


def main():
    """
    Main function to fetch and save job listings with pagination.
    """
    all_job_listings = []

    if not os.path.exists(os.path.join(output_folder,params['location'])):
        os.makedirs(os.path.join(output_folder,params['location']))

    for start in range(0, 300, 10):  # Adjust range as needed to fetch more pages
        params['start'] = start
        job_listings, response = fetch_job_listings(params)
        if job_listings:
            all_job_listings.extend(job_listings)
            save_response_as_json(response, start)
        else:
            logging.warning("No more job listings found. Stopping the fetch.")
            break
    save_to_csv(all_job_listings)

if __name__ == '__main__':
    main()
