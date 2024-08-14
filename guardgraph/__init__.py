import os
import base64
from io import BytesIO
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
from wtforms import StringField, SubmitField, IntegerRangeField, BooleanField, FloatField, IntegerField, HiddenField, SelectField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, Optional
import urllib.parse
import folium
from PIL import Image
import pandas as pd
from pygbif import species, occurrences
from guardgraph.gbif import cube_query

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
app.config['SECRET_KEY'] = os.urandom(12).hex() # to allow csrf forms
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # max 50MB upload
db = SQLAlchemy()
fef = FEFset(frontend='bootstrap4')
fef.nav_menu.append(
    {'name':'Home', 'url':'/'}
)
fef.nav_menu.append(
    {'name':'Species interactions', 'url':'/species/interactions/log'}
)
fef.nav_menu.append(
    {'name':'Case study cubes', 'url':'/species/interactors/guarden/cube'}
)
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
    thumbnail = db.Column(db.String)
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

class GuardenCaseForm(FlaskForm):
    case_study = SelectField(
        'Case study',
        choices=[
            'France','Madagascar','Greece','Spain','Cyprus'
        ]
    )
    cubes = SelectField(
        'Cubes',
        choices=[
            'Only species of interest',
            'Including first-order interacting species'
        ]
    )
    species_file = FileField('Species of interest', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV only!')
    ])
    header = BooleanField('Header in CSV?', default=True)
    column = IntegerField('Column number with species names (0-indexed)', default=0)
    submit_button = SubmitField('Submit')

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/species/interactions/log', methods=['GET','POST'])
def ix_log():
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
        image = Image.open(os.path.join(app.instance_path, 'ix/' + ix_filename))
        MAX_SIZE = (70, 70)
        image.thumbnail(MAX_SIZE)
        imgbytes = BytesIO()
        image.save(imgbytes, format='png')
        imgbytes.seek(0)                                                                                                     
        img_html = '<img src="data:image/png;base64,{}">'.format(
            base64.b64encode(imgbytes.read()).decode('UTF-8')
        )
        ix.thumbnail = img_html
        
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
                <tr><th>Species 1</th><td>{ix.species_1 or 'Unknown'}</td></tr>
                <tr><th>Species 2</th><td>{ix.species_2 or 'Unknown'}</td></tr>
                <tr><th>Interaction<th><td>{ix.ix_type}</td></tr>
            </table>{ix.thumbnail}"""
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
    input_data = request.get_json()
    species_list = input_data['species']
    speciesKeyList = prep_speciesKey_list(species_list)
    cube_job_id = cube_query(
        input_data['email'], input_data['gbif_user'],
        input_data['gbif_pwd'], input_data['polygon'],
        speciesKeyList
    )
    return jsonify({'cube_job_id': cube_job_id})

def prep_speciesKey_list(species_list, with_interactors=True):
    if with_interactors:
        interactors = query_interactions(species_list)
    else: interactors = []
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
    return speciesKeyList

@app.route('/species/interactors/guarden/cube', methods=['GET','POST'])
def guarden_cube():
    form=GuardenCaseForm()
    if form.validate_on_submit():
        species_df = pd.read_csv(
            form.species_file.data,
            header=1 if form.header.data else None
        )
        species_list = list(
            species_df[species_df.columns[form.column.data]]
        )
        speciesKeyList = prep_speciesKey_list(
            species_list,
            with_interactors=form.cubes.data != 'Only species of interest'
        )
        cube_job_id = case_study_cube(
            form.case_study.data,
            speciesKeyList
        )
        return redirect(f"/species/interactors/cube/{cube_job_id}")
    return render_template('gsc.html', form=form)

@app.route('/species/interactors/cube/<cube_job_id>', methods=['GET'])
def download_interactors_cube(cube_job_id):
    from guardgraph.gbif import download_cube
    try:
        download_cube(
            cube_job_id,
            prefix=os.path.join(app.instance_path, 'cubes')+'/',
            wait=False
        )
        return render_template('cd.html', cube_job_id=cube_job_id)
    except Exception as e:
        print(e)
        return render_template('cd.html')

def get_case_study_polygon(case_study_name, polygon_simplifier=1000):
    import tarfile
    from shapely import wkt, Polygon, MultiPolygon
    case_study_polygon_files = {
        'France': 'cbnmed.wkt.txt',
        'Madagascar': 'pnrn.wkt.txt',
        'Greece': 'greece.wkt.txt',
        'Spain': 'barcelona.wkt.txt',
        'Cyprus': 'cyprus.wkt.txt'
    }
    case_study_polygon_file = case_study_polygon_files[
        case_study_name
    ]
    case_studies = tarfile.open(
        os.path.join(
            os.path.dirname(__file__), 'static/data/guarden_case_studies.tar.xz'
        )
    )
    case_study_wkt = case_studies.extractfile(
        case_studies.getmember(case_study_polygon_file)
    ).read()
    case_study_polygon = wkt.loads(case_study_wkt)
    if polygon_simplifier:
        case_study_polygon = case_study_polygon.simplify(
            polygon_simplifier, preserve_topology=False
        )
    # Transform to WGS84 for GBIF
    from shapely import wkt, Polygon, MultiPolygon
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    def transform_polygon(polygon, transformer):
        if isinstance(polygon, Polygon):
            return Polygon([
                transformer.transform(x, y)
                for x, y in polygon.exterior.coords
            ])
        elif isinstance(polygon, MultiPolygon):
            return MultiPolygon([Polygon([
                transformer.transform(x, y)
                for x, y in pc.exterior.coords
            ]) for pc in polygon.geoms])

    case_study_polygon = transform_polygon(
        case_study_polygon, transformer
    )    
    return case_study_polygon
        
def case_study_cube(case_study_name, speciesKeyList, polygon_simplifier=1000):
    case_study_polygon = get_case_study_polygon(
        case_study_name, polygon_simplifier
    )
    cube_job_id = cube_query(
        'christophe.vanneste@plantentuinmeise.be',
        'cvanneste',
        open('gbif_pwd','rt').read().strip(),
        case_study_polygon.wkt,
        speciesKeyList
    )
    return cube_job_id

@app.route('/casestudy/<case_study_name>', methods=['GET'])
def visualize_case_study(case_study_name):
    from shapely import to_geojson
    polygon = get_case_study_polygon(case_study_name.capitalize())
    center_coords = [polygon.centroid.y, polygon.centroid.x]
    tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png'
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>'
    map = folium.Map(location=center_coords, tiles=tiles, attr=attr, zoom_start=11)
    folium.Marker(location=center_coords, icon=folium.Icon(color='darkgreen', icon='seedling', prefix='fa'),
        popup=case_study_name
    ).add_to(map)
    # Add case study polygon
    geo_j = folium.GeoJson(
        data=to_geojson(polygon),
        style_function=lambda x: {"fillColor": "orange"}
    )
    folium.Popup(case_study_name).add_to(geo_j)
    geo_j.add_to(map)
    # set the iframe width and height
    map.get_root().width = "100%" #"800px"
    map.get_root().height = "600px"
    iframe = map.get_root()._repr_html_()
    return render_template(
        "cs_vis.html", iframe=iframe
    )
