import os
import datetime
from flask import Flask, jsonify, request
from guardgraph.graph import InteractionsGraph

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello Interactions Seeker'

@app.route('/init')
def init_server():
    # This entrypoint only runs once and should be called when the db is ready
    if os.path.exists('INITIATED'):
        return 'Already initiated'
    else:
        with open('INITIATED','wt') as fout:
            fout.write(str(datetime.datetime.now()))
        ig = InteractionsGraph()
        ig.set_random_password()
        # Test and return
        return str(ig.relationships)

@app.route('/species', methods=['POST'])
def describe_species():
    species = request.get_json()
    ig = InteractionsGraph()
    data = {
        s: ig.run_query(
            'MATCH (n:species)-[r]-() WHERE n.name STARTS WITH "'
            +s.split()[0]+'" RETURN COUNT(r)'
        )
        for s in species
    }
    return jsonify(data)

@app.route('/species/interactions', methods=['POST'])
def get_interactions():
    species = request.get_json()
    data = query_interactions(species)
    return jsonify(data)

def query_interactions(species):
    ig = InteractionsGraph()
    data = {
        s: ig.run_query(
            'MATCH (n:species)-[r]-(m:species) WHERE n.name STARTS WITH "'
            +s.split()[0]+'" RETURN DISTINCT m'
        )
        for s in species
    }
    return data


@app.route('/species/interactors/cube', methods=['POST'])
def get_interactors_cube():
    from pygbif import species, occurrences
    from guardgraph.gbif import cube_query
    input_data = request.get_json()
    species_list = input_data['species']
    interactors = query_interactions(species)
    speciesKeyList = set([
        (species.name_suggest(
            s, limit=1
        ) or [{'speciesKey':None}]
         )[0]['speciesKey'] for s in species_list
    ]+[
        (species.name_suggest(
            i['m']['name'], limit=1
        ) or [{'speciesKey':None}]
         )[0]['speciesKey']
        for s in interactors
        for i in interactors[s]
    ])
    try:
        speciesKeyList.remove(None)
        print('GBIF unknown species were present')
    except KeyError:
        print('All species known')
    speciesKeyList = [str(s) for s in speciesKeyList]
    cube_job_id = cube_query(
        input_data['email'], input_data['gbif_user'],
        input_data['gbif_pwd'], input_data['polygon'],
        speciesKeyList
    )
    return jsonify({'cube_job_id': cube_job_id})

@app.route('/species/interactors/cube/<cube_job_id>', methods=['GET'])
def download_interactors_cube(cube_job_id):
    from guardgraph.gfib import download_cube
    download_cube(cube_job_id, prefix='/data/cubes')
    return 'TODO pass download link'
