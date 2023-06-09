import re
import os
import json
import gzip
import requests
import glob
import subprocess as sp

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

class Sorter(object):
    def __init__(self, output_dir, bucket_hash_len=4, compressed_files=True):
        "bucket_hash_len>4 too many files to handle"
        self.compressed = compressed_files
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.hash_len = bucket_hash_len
        self.buckets = {}

    def buckit(self, key, line):
        h = str(hash(key))
        try: self.buckets[h[:self.hash_len]].write(h+'\t'+line)
        except KeyError:
            if self.compressed:
                self.buckets[h[:self.hash_len]] = gzip.open(
                    os.path.join(self.output_dir,'b'+h[:self.hash_len]+'.bckt.gz'),
                    'wt'
                )
            else:
                self.buckets[h[:self.hash_len]] = open(
                    os.path.join(self.output_dir,'b'+h[:self.hash_len]+'.bckt'),
                    'wt'
                )
            self.buckets[h[:self.hash_len]].write(h+'\t'+line)

    def close(self):
        for f in self.buckets.values():
            f.close()

    def sort_buckets(self, output_filename, progress_bar=False, remove_hash=False):
        if remove_hash:
            # TODO functionality to remove hash from lines after sorting
            raise NotImplementedError
        buckets = glob.glob(os.path.join(self.output_dir,'*.bckt*'))
        if progress_bar:
            import tqdm
        count = 0
        with (gzip.open if self.compressed else open)(
            output_filename, 'wt'
        ) as output:
            for bucket in (tqdm.tqdm(buckets) if progress_bar else buckets):
                if self.compressed:
                    sorted_bucket = bucket[:3]+'_sorted'
                    sp.call(f"zcat {bucket} | sort > {sorted_bucket}", shell=True)
                else:
                    sorted_bucket = bucket+'_sorted'
                    sp.call(f"sort {bucket} > {sorted_bucket}", shell=True)
                #with (gzip.open if self.compressed else open)('bucket, 'rt') as b:
                    # Python sort gets killed for files > 10 MB
                    # It would be quicker to still use this code for smaller buckets
                    #bucket_content = [
                    #    (l[:l.index('\t')],l)
                    #    for l in b
                    #]
                    #bucket_content.sort(key=lambda x:x[0])
                    #for h,l in bucket_content:
                with open(sorted_bucket, 'rt') as b:
                    for l in b:
                        output.write(l)
                        count += 1
                os.remove(bucket)
                os.remove(sorted_bucket)
        return count
