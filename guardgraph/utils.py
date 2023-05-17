import re
import os
import requests

def download_file(url, destination_folder='/data', check_file=True):
    headers = requests.head(url, allow_redirects=True).headers
    if check_file:
        ct = headers['Content-Type'].lower()
        assert 'html' not in ct and 'text' not in ct
    #content_size = headers['Content-Length']
    filere = re.compile(r'filename=(.+)')
    filename = filere.search(
        headers['Content-Disposition']
    ).groups()[0]
    filename = os.path.join(destination_folder, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as out:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                out.write(chunk)
    return filename
