import os
import datetime
from flask import Flask, jsonify, request, render_template, redirect
from guardgraph.graph import InteractionsGraph
from flask_fefset import FEFset
from flask_uxfab import UXFab
from flask_sqlalchemy import SQLAlchemy
from flask_iam import IAM
# IX Form
from flask_login import current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerRangeField, BooleanField, FloatField, IntegerField, HiddenField
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, Optional
import urllib.parse
import folium

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
app.config['SECRET_KEY'] = os.urandom(12).hex() # to allow csrf forms
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # max 100MB upload
db = SQLAlchemy()
fef = FEFset(frontend='bootstrap4')
fef.nav_menu.append({'name':'Species interactions','url':'/'})
fef.settings['brand_name'] = 'GUARDEN-IX'
fef.settings['logo_url'] = '/static/images/guarden_logo.png'
fef.init_app(app)
db.init_app(app)
uxf = UXFab()
uxf.init_app(app)
iam = IAM(db)
iam.init_app(app)

# Data models
class IXObservation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    accuracy = db.Column(db.Float)
    species_1 = db.Column(db.String)
    species_2 = db.Column(db.String)
    ix_type = db.Column(db.String)
    img_species_1 = db.Column(db.String)
    img_species_2 = db.Column(db.String)
    img_ix12 = db.Column(db.String)
    datetime = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship('User')

class IXOForm(FlaskForm):
    latitude = FloatField('Latitude of tasting')
    longitude = FloatField('Longitude of tasting')
    accuracy = HiddenField(default=-1)
    img_ix12 = FileField('Interaction image', validators=[InputRequired()])
    ix_type = StringField('Interaction type', validators=[InputRequired()])
    img_species_1 = FileField('Species 1 image')
    species_1 = StringField('Species 1')
    img_species_2 = FileField('Species 2 image')
    species_2 = StringField('Species 2')
    submit_button = SubmitField('Submit Interaction')
    
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET','POST'])
def index():
    mbg_coords = [50.9211519, 4.3317191]
    form=IXOForm()
    if form.validate_on_submit():
        coords = [form.latitude.data, form.longitude.data]
        ix = IXObservation()
        form.populate_obj(ix)
        ix.datetime = datetime.datetime.now()
        if current_user.is_authenticated:
            ix.user_id = current_user.id
        ix_filename = secure_filename(
            form.img_ix12.data.filename
        )
        # TODO check potential overwriting
        form.img_ix12.data.save(
            os.path.join(app.instance_path, 'ix/' + ix_filename)
        )
        ix.img_ix12 = ix_filename
        if form.img_species_1.data:
            s1_filename = secure_filename(
                form.img_species_1.data.filename
            )
            # TODO check potential overwriting
            form.img_species_1.data.save(
                os.path.join(app.instance_path, 'ix/' + s1_filename)
            )
            ix.img_species_1 = s1_filename
        else: ix.img_species_1 = None
        if form.img_species_2.data:
            s2_filename = secure_filename(
                form.img_species_2.data.filename
            )
            # TODO check potential overwriting
            form.img_species_2.data.save(
                os.path.join(app.instance_path, 'ix/' + s2_filename)
            )
            ix.img_species_2 = s2_filename
        else: ix.img_species_2 = None
        db.session.add(ix)
        db.session.commit()
        form = None
    else:
        coords = mbg_coords
        form_marker = None
    tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png'
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>'
    map = folium.Map(location=coords, tiles=tiles, attr=attr, zoom_start=11)
    #folium.Marker(coords, popup=popup_msg).add_to(map)
    #https://fontawesome.com/search?m=free&o=r
    folium.Marker(location=mbg_coords, icon=folium.Icon(color='darkgreen', icon='seedling', prefix='fa'),
        popup='A botanic garden with a lot of interactions'
    ).add_to(map)
    for ix in IXObservation.query.all():
        folium.Marker(
            (ix.latitude,ix.longitude),
            icon=folium.Icon(color='darkbrown', icon='seedling', prefix='fa'),
            popup=f"""<table>
                <tr><th>Species 1</th><td>{ix.species_1}</td></tr>
                <tr><th>Species 2</th><td>{ix.species_2}%</td></tr>
                <tr><th>Interaction<th><td>{ix.ix_type}</td></tr>
            </table>"""
        ).add_to(map)
    # set the iframe width and height
    map.get_root().width = "100%" #"800px"
    map.get_root().height = "600px"
    iframe = map.get_root()._repr_html_()
    return render_template(
        "ix.html", form=form, iframe=iframe
    )

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
