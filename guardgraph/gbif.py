"Copied from https://github.com/AgentschapPlantentuinMeise/irowine/blob/main/interactions/ixro/query.py"

import os
import json
import time
from urllib.request import urlretrieve
import requests

query_template = '''
{{
  "sendNotification": true,
  "notificationAddresses": [
    "{email}"
  ],
  "format": "SQL_TSV_ZIP",
  "sql": "SELECT \
    PRINTF('%04d-%02d', \\"year\\", \\"month\\") AS yearMonth,\
    GBIF_EEARGCode(\
      1000,\
      decimalLatitude,\
      decimalLongitude,\
      COALESCE(coordinateUncertaintyInMeters, 1000)\
      ) AS eeaCellCode,\
    familyKey,\
    family,\
    speciesKey,\
    species,\
    datasetKey,\
    COALESCE(sex, 'NOT_SUPPLIED') AS sex,\
    COALESCE(occurrence.lifestage.concept, 'NOT_SUPPLIED') AS lifestage, \
    COUNT(*) AS occurrences\
  FROM\
    occurrence\
  WHERE\
    GBIF_Within('{geometry}', decimalLatitude, decimalLongitude) = True\
    AND occurrenceStatus = 'PRESENT'\
    AND \\"year\\" >= 2000\
    AND hasCoordinate = TRUE\
    AND speciesKey IN ({speciesKeyList})\
    AND NOT ARRAY_CONTAINS(issue, 'ZERO_COORDINATE')\
    AND NOT ARRAY_CONTAINS(issue, 'COORDINATE_OUT_OF_RANGE')\
    AND NOT ARRAY_CONTAINS(issue, 'COORDINATE_INVALID')\
    AND NOT ARRAY_CONTAINS(issue, 'COUNTRY_COORDINATE_MISMATCH')\
    AND \\"month\\" IS NOT NULL\
  GROUP BY\
    yearMonth,\
    datasetKey,\
    eeaCellCode,\
    familyKey,\
    family,\
    speciesKey,\
    species,\
    sex,\
    lifestage\
  ORDER BY\
    yearMonth DESC,\
    eeaCellCode ASC,\
    speciesKey ASC;\
    "
}}
'''

def cube_query(email, gbif_user, gbif_pwd, geometry, speciesKeyList, query_template=query_template):
    query = query_template.format(
        email=email, geometry=geometry,
        speciesKeyList=', '.join(speciesKeyList)
    )
    #with open('query.json','wt') as qf:
    #    qf.write(query)
    headers = {
        'Content-type': 'application/json',
        #'Accept': 'text/plain'
    }
    r_valid = requests.post(
        'https://api.gbif.org/v1/occurrence/download/request/validate',
        data=query, headers=headers
    )
    if r_valid.status_code > 201:
        raise Exception(r_valid.status_code)
    r_cube = requests.post(
        'https://api.gbif.org/v1/occurrence/download/request',
        data=query, headers=headers,
        auth = (gbif_user, gbif_pwd)
    )
    if r_cube.status_code > 201:
        raise Exception(r_cube.status_code)
    return r_cube.text # cube job id

def download_cube(cube_job_id, prefix):
    while (r:=json.loads(requests.get(
        f"https://api.gbif.org/v1/occurrence/download/{cube_job_id}"
    ).text))['status'] == 'RUNNING':
        time.sleep(60)
    if r['status'] != 'SUCCEEDED':
        raise Exception(r['status'])
    urlretrieve(
        r['downloadLink'],
        prefix+r['downloadLink'][r['downloadLink'].rindex('/')+1:]
    )

#curl --include --header "Content-Type: application/json" --data @query.json https://api.gbif.org/v1/occurrence/download/request/validate

#curl --include --user 'maxime_ryckewaert':'password' --header "Content-Type: application/json" --data @query.json https://api.gbif.org/v1/occurrence/download/request
