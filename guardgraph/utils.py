import re
import os
import json
import requests

def download_file(url, destination_folder='/data', check_file=True, progress_bar=False):
    headers = requests.head(url, allow_redirects=True).headers
    if check_file:
        ct = headers['Content-Type'].lower()
        assert 'html' not in ct and 'text' not in ct
    content_size = headers['Content-Length']
    filere = re.compile(r'filename=(.+)')
    filename = filere.search(
        headers['Content-Disposition']
    ).groups()[0]
    filename = os.path.join(destination_folder, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as out:
            chunk_size = 8192
            if progress_bar:
                from tqdm import tqdm
                from math import ceil
                chunk_iter = tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=ceil(int(content_size)/chunk_size),
                    unit='mb', unit_scale=chunk_size/1024**2
            )
            else: chunk_iter = r.iter_content(chunk_size=chunk_size)
            for chunk in chunk_iter: 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                out.write(chunk)
    return filename

def globi_web_cypher(query):
    """Reference: https://github.com/ropensci/rglobi/blob/HEAD/R/rglobi.R
    Available labels: Taxon (41818247), Reference (11023639)
    """
    response = requests.post(
        'https://neo4j.globalbioticinteractions.org/db/data/cypher',
        data=json.dumps(
            {'query':query}
        ), headers={
            'Accept':'application/json',
            'Content-Type':'application/json'
        }
    )
    return json.loads(response.content)
