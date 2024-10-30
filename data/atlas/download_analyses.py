import time
import requests
from tqdm import tqdm
#import pdb

def download_url(url, retries=3, backoff_factor=0.5):
    """Download the content from a URL with retries.

    Args:
        url (str): The URL to download.
        retries (int): Number of retries if the download fails. Default is 3.
        backoff_factor (float): Factor by which to multiply the delay between retries. Default is 0.5.

    Returns:
        response: The response object from requests if successful, None otherwise.
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries}: Downloading {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
            print("Download successful!")
            return response
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    print("All retries failed.")
    return None

with open('2022_06_13_ATLAS_pdb.txt','r') as pdb_codes_file:
    pdb_codes = pdb_codes_file.readlines()
    pdb_codes = [p.strip() for p in pdb_codes]

# Example usage:
url_base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/"

for pdb_code in tqdm(pdb_codes):
    url = url_base + pdb_code
    #pdb.set_trace()
    response = download_url(url)

    if response.status_code == 200:
        # Open a file in binary write mode
        with open('zipped_files/'+pdb_code+'_analysis.zip', 'wb') as fileout:
            # Write the content of the response to the file
            fileout.write(response.content)