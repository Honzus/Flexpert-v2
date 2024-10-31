import time
import requests
from tqdm import tqdm
import zipfile
import io
import os
import yaml

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

# Option 1: Direct unzip from response content
def save_and_unzip_response(response, extract_path):
    """
    Directly unzip response content to a directory without saving zip file
    
    Args:
        response: requests response object
        extract_path (str): Directory to extract files to
    """
    # Create a BytesIO object from the response content
    zip_bytes = io.BytesIO(response.content)
    
    # Extract using zipfile
    with zipfile.ZipFile(zip_bytes) as zip_ref:
        zip_ref.extractall(extract_path)

if __name__== "__main__":
    pdb_codes_path = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['pdb_codes_path']
    out_dir = yaml.load(open('configs/data_config.yaml', 'r'), Loader=yaml.FullLoader)['atlas_out_dir']

    os.makedirs(out_dir, exist_ok=True)

    # Read the list of PDB codes from the file
    with open(pdb_codes_path,'r') as pdb_codes_file:
        pdb_codes = pdb_codes_file.readlines()
        pdb_codes = [p.strip() for p in pdb_codes]

    # Example usage:
    url_base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/"

    for pdb_code in tqdm(pdb_codes):
        url = url_base + pdb_code
        #pdb.set_trace()
        response = download_url(url)

        if response and response.status_code == 200:
            # Extract directly to a directory
            extract_path = os.path.join(out_dir, pdb_code)
            os.makedirs(extract_path, exist_ok=True)
            save_and_unzip_response(response, extract_path)