import os
import datetime
from flask import Flask

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
        from guardgraph.graph import InteractionsGraph
        with open('INITIATED','wt') as fout:
            fout.write(str(datetime.datetime.now()))
        ig = InteractionsGraph()
        ig.set_random_password()
        # Test and return
        return str(ig.relationships)
