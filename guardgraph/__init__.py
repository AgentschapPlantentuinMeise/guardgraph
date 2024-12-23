import os
import base64
from io import BytesIO
import datetime
from flask import Flask, jsonify, request, render_template, redirect, abort
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
from shapely import wkt
import folium
from PIL import Image
import pandas as pd
from pygbif import species, occurrences
from guardgraph.gbif import cube_query
from guardgraph import tasks
from celery.result import AsyncResult

def create_app(config_filename=None):
    app = Flask(__name__)

    # Config
    #app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(app.instance_path,'shared/db.sqlite')}"
    #app.config["SQLALCHEMY_DATABASE_URI"] = f"mariadb+pymysql://guardin:{os.environ.get('MARIADB_PASSWORD')}@db:3306/"
    app.config['SECRET_KEY'] = os.urandom(12).hex() # to allow csrf forms
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # max 50MB upload
    app.config.from_mapping(
        CELERY=dict(
            broker_url=os.environ.get("CELERY_BROKER_URL"),#'sqla+sqlite:////tmp/celery.db'
            result_backend=f"db+sqlite:///{os.path.join(app.instance_path,'shared/celery.db')}",
            #os.environ.get("CELERY_RESULT_BACKEND", "rpc://"),
            task_ignore_result=True,
        ),
    )
    if config_filename:
        app.config.from_pyfile(config_filename)

    # Flask extensions
    db = SQLAlchemy()
    fef = FEFset(frontend='bootstrap4')
    fef.nav_menu.append(
        {'name':'Home', 'url':'/'}
    )
    fef.nav_menu.append(
        {'name':'Case study analysis', 'url':'/casestudy/analysis'}
    )
    fef.nav_menu.append(
        {'name':'Case study cubes', 'url':'/species/interactors/guarden/cube'}
    )
    fef.nav_menu.append(
        {'name':'Species interactions', 'url':'/species/interactions/log'}
    )
    fef.settings['brand_name'] = 'GUARDEN-IX'
    fef.settings['logo_url'] = '/static/images/guarden_logo.png'
    fef.init_app(app)
    db.init_app(app)
    uxf = UXFab()
    uxf.init_app(app)
    iam = IAM(db)
    iam.init_app(app)
    celery_app = tasks.celery_init_app(app)

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
        img_ix1 = db.Column(db.String)
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
        img_ix1 = FileField('Post interaction image (same framing as inter)', validators=[InputRequired()])
        img_species_1 = FileField('Species 1 image')
        species_1 = StringField('Species 1')
        img_species_2 = FileField('Species 2 image (optional)')
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
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500
    
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')
    
    @app.route('/species/interactions/log/<int:cell>', methods=['GET','POST'])
    @app.route('/species/interactions/log', methods=['GET','POST'])
    def ix_log(cell=None):
        from guardgraph.mbg import mbg_polygon
        from intercubos.gridit import Grid
        import folium
        from folium.plugins import LocateControl
        from shapely import to_geojson
        from geopandas import GeoSeries
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
            # Interaction image
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
    
            # Post interaction image
            pix_filename = secure_filename(
                form.img_ix1.data.filename
            )
            # TODO check potential overwriting
            form.img_ix1.data.save(
                os.path.join(app.instance_path, 'ix/' + pix_filename)
            )
            ix.img_ix1 = pix_filename
    
            # Species 1
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
            # Species 2
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
            form_marker = None
        tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png'
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>'
        coords = (mbg_polygon.centroid.y, mbg_polygon.centroid.x)
        map = folium.Map(location=coords, tiles=tiles, attr=attr, zoom_start=11)
        # Make grid
        local_epsg = 3857 # for area respecting rotations # 4087
        mbg_centroid = GeoSeries(mbg_polygon.centroid, crs=4326)
        grid = Grid(*mbg_polygon.bounds, stepsize=50) # stepsize in meters
        # TODO rotate angle in grid init
        rotated_grid = grid.grid.copy(deep=True)
        #rotated_grid['geometry'] = rotated_grid['geometry'].rotate(
        #    angle=10, origin=mbg_polygon.centroid
        #)
        rotated_grid['geometry'] = rotated_grid.to_crs(
            epsg=local_epsg
        )['geometry'].rotate(
            angle=25,
            origin=mbg_centroid.to_crs(epsg=local_epsg).iloc[0]
        ).to_crs(epsg=4326)
        #grid.grid['geometry'] = grid.grid['geometry'].rotate(
        #    angle=45, origin='centroid'
        #)
        folium.GeoJson(rotated_grid).add_to(map)
        folium.GeoJson(
            data=to_geojson(mbg_polygon),
            style_function=lambda x: {"fillColor": "orange"}
        ).add_to(map)
        if cell is not None:
            folium.GeoJson(
                data=to_geojson(rotated_grid.iloc[cell].geometry),
                style_function=lambda x: {"fillColor": "red", "borderColor": "red"}
            ).add_to(map)
        
        #folium.Marker(coords, popup=popup_msg).add_to(map)
        #https://fontawesome.com/search?m=free&o=r    
        folium.Marker(location=coords, icon=folium.Icon(color='darkgreen', icon='seedling', prefix='fa'),
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
        LocateControl(
            auto_start=True,
            locateOptions={'enableHighAccuracy': True, 'watch': True}
        ).add_to(map)
        iframe = map.get_root()._repr_html_()
        return render_template(
            "ix.html", form=form, iframe=iframe
        )
    
    @app.route('/init')
    def init_server():
        # This entrypoint only runs once and should be called when the db is ready
        if os.path.exists('INITIATED'):
            ig = InteractionsGraph()
            return f"Already initiated with {ig.relationships}"
        else:
            with open('INITIATED','wt') as fout:
                fout.write(str(datetime.datetime.now()))
            ig = InteractionsGraph(password='neo4j')
            ig.set_password(os.environ.get('NEO4J_CREDENTIAL'))
            # Test and return
            return str(ig.relationships)

    @app.route('/genus', methods=['POST'])
    def describe_genus():
        """
        Example:
    
            import requests
            url = 'https://www.guardin.net/species'
            # or local docker compose: 'http://web:5000/species'
            requests.post(url, json=[
              'Accipiter nisus','Acrocephalus arundinaceus'
            ]).json()
    
        """
        geni = request.get_json()
        ig = InteractionsGraph()
        data = {
            g: ig.run_query(
                'MATCH (n:species)-[r]-() WHERE n.name STARTS WITH "'
                +g.split()[0]+'" RETURN COUNT(r) AS count'
            )
            for g in geni
        }
        return jsonify(data)
    
    @app.route('/genus/interactions', methods=['POST'])
    def get_genus_interactions():
        geni = request.get_json()
        data = tasks.query_interactions(geni)
        return jsonify(data)

    @app.route('/genus/interaction/citations', methods=['POST'])
    def get_genus_interaction_citations():
        geni = request.get_json()
        data = tasks.query_interaction_citations(geni)
        return jsonify(data)
    
    @app.route('/genus/inter2x', methods=['POST'])
    def get_2x_genus_interactions():
        geni = request.get_json()
        data = tasks.query_interactions(geni, second_order=True)
        return jsonify(data)
    
    @app.route('/species', methods=['POST'])
    def describe_species():
        """
        Example:
    
            import requests
            url = 'https://www.guardin.net/species'
            # or local docker compose: 'http://web:5000/species'
            requests.post(url, json=[
              'Accipiter nisus','Acrocephalus arundinaceus'
            ]).json()
    
        """
        species = request.get_json()
        ig = InteractionsGraph()
        data = {
            s: ig.run_query(
                'MATCH (n:species)-[r]-() WHERE n.name STARTS WITH "'
                +' '.join(s.split()[:2])+'" RETURN COUNT(r) AS count'
            )
            for s in species
        }
        return jsonify(data)
    
    @app.route('/species/interactions', methods=['POST'])
    def get_interactions():
        species = request.get_json()
        data = tasks.query_interactions(species)
        return jsonify(data)

    @app.route('/species/interaction/citations', methods=['POST'])
    def get_interaction_citations():
        species = request.get_json()
        data = tasks.query_interaction_citations(species)
        return jsonify(data)
    
    @app.route('/species/inter2x', methods=['POST'])
    def get_2x_interactions():
        species = request.get_json()
        data = tasks.query_interactions(species, second_order=True)
        return jsonify(data)
    
    @app.route('/species/embedding', methods=['POST'])
    def get_embedding():
        species = request.get_json()
        data = tasks.interaction_embedding(species)
        return jsonify(data)
    
    @app.route('/species/interactors/cube', methods=['POST'])
    def get_interactors_cube(task_id=None):
        # TODO check species list
        input_data = request.get_json()
        cube_job_id = cube_query(
            input_data['email'], input_data['gbif_user'],
            input_data['gbif_pwd'], input_data['polygon'],
            input_data['species']
        )
        return jsonify({'cube_job_id': cube_job_id})
    
    @app.route('/species/interactors/guarden/cube', methods=['GET','POST'])
    @app.route('/species/interactors/guarden/cube/<task_id>', methods=['GET'])
    def guarden_cube(task_id=None):
        if task_id:
            result = AsyncResult(task_id)
            if not result.ready():
                return render_template('refresh.html')
            elif not result.successful():
                abort(500)
            else: return redirect(f"/species/interactors/cube/{result.result}")
        else:
            form=GuardenCaseForm()
            if form.validate_on_submit():
                species_df = pd.read_csv(
                    form.species_file.data,
                    header=1 if form.header.data else None
                )
                species_list = list(
                    species_df[species_df.columns[form.column.data]]
                )
                with_interactors = form.cubes.data != 'Only species of interest'
                task_id = tasks.case_study_cube.delay(
                    form.case_study.data, species_list,
                    with_interactors=with_interactors
                )
                return redirect(f"/species/interactors/guarden/cube/{task_id}")
            return render_template('gsc.html', title='Case study cube', form=form)
    
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
    
    @app.route('/casestudy/<case_study_name>', methods=['GET'])
    def prep_visualize_case_study(case_study_name):
        result = tasks.get_case_study_polygon.delay(
                    case_study_name.capitalize()
        )
        return redirect(f"/casestudy/{case_study_name}/{result.id}")
    
    
    @app.route('/casestudy/<case_study_name>/<case_study_task_id>', methods=['GET'])
    def visualize_case_study(case_study_name,case_study_task_id):
        from shapely import to_geojson
        result = AsyncResult(case_study_task_id)
        if not result.ready():
            return render_template('refresh.html')
        elif not result.successful():
            abort(500)
        polygon = wkt.loads(result.result)
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
    
    @app.route('/casestudy/analysis', methods=['GET','POST'])
    @app.route('/casestudy/analysis/<task_id>', methods=['GET'])
    def guarden_case_analysis(task_id=None):
        if task_id:
            result = AsyncResult(task_id)
            if not result.ready():
                return render_template('refresh.html')
            elif not result.successful():
                abort(500)
            else:
                # Outputs
                outputs = {}
                for output in result.result['files']:
                    if result.result['files'][output].endswith('.svg'):
                        with open(result.result['files'][output], 'rt') as f:
                            outputs[output] = f.read()
                    elif result.result['files'][output].endswith('.csv'):
                        outputs[output] = pd.read_csv(result.result['files'][output]).to_html()
                #svg_bytes = io.BytesIO()
                #fig.savefig(svg_bytes, format='svg')
                #svg_bytes.getvalue().decode()
                return render_template(
                    "analysis.html",
                    title = result.result['title'],
                    outputs = outputs
                )
    
                print(result.result)
                return 'works'
            #redirect(f"/casestudy/analysis/{result.result}")
        else:
            form=GuardenCaseForm()
            del form.cubes
            if form.validate_on_submit():
                species_df = pd.read_csv(
                    form.species_file.data,
                    header=1 if form.header.data else None
                )
                species_list = list(
                    species_df[species_df.columns[form.column.data]]
                )
                task_id = tasks.case_study_interactions.delay(
                    form.case_study.data, species_list
                )
                return redirect(f"/casestudy/analysis/{task_id}")
            return render_template('gsc.html', title='Case study analysis', form=form)
    
    @app.route('/cube/analysis/<cube_id>', methods=['GET','POST'])
    @app.route('/cube/analysis/<cube_id>/<task_id>', methods=['GET'])
    def guarden_cube_analysis(cube_id,task_id=None):
        if task_id:
            result = AsyncResult(task_id)
            if not result.ready():
                return render_template('refresh.html')
            elif not result.successful():
                abort(500)
            else:
                # Output
                iframe = result.result
                return render_template(
                    "cs_vis.html", iframe=iframe
                )
        else:
            task_id = tasks.analyze_casestudy_data.delay(
                os.path.join(app.instance_path,f'cubes/{cube_id}.zip')
            )
            return redirect(f"/cube/analysis/{cube_id}/{task_id}")
        
    @app.route('/mbg/grid', methods=['GET'])
    @app.route('/mbg/grid/<int:cell>', methods=['GET'])
    def mbg_interactions(cell=None):
        from guardgraph.mbg import mbg_polygon
        from intercubos.gridit import Grid
        import folium
        from folium.plugins import LocateControl
        from shapely import to_geojson
        from geopandas import GeoSeries
        local_epsg = 3857 # for area respecting rotations # 4087
        mbg_centroid = GeoSeries(mbg_polygon.centroid, crs=4326)
        grid = Grid(*mbg_polygon.bounds, stepsize=50) # stepsize in meters
        # TODO rotate angle in grid init
        rotated_grid = grid.grid.copy(deep=True)
        #rotated_grid['geometry'] = rotated_grid['geometry'].rotate(
        #    angle=10, origin=mbg_polygon.centroid
        #)
        rotated_grid['geometry'] = rotated_grid.to_crs(
            epsg=local_epsg
        )['geometry'].rotate(
            angle=25,
            origin=mbg_centroid.to_crs(epsg=local_epsg).iloc[0]
        ).to_crs(epsg=4326)
        #grid.grid['geometry'] = grid.grid['geometry'].rotate(
        #    angle=45, origin='centroid'
        #)
        (c_lat, c_lon) = (mbg_polygon.centroid.y, mbg_polygon.centroid.x)
        m = folium.Map(
            location=(c_lat,c_lon), prefer_canvas=True, zoom_start=12
        )
        folium.GeoJson(rotated_grid).add_to(m)
        folium.GeoJson(
            data=to_geojson(mbg_polygon),
            style_function=lambda x: {"fillColor": "orange"}
        ).add_to(m)
        if cell is not None:
            folium.GeoJson(
                data=to_geojson(rotated_grid.iloc[cell].geometry),
                style_function=lambda x: {"fillColor": "red", "borderColor": "red"}
            ).add_to(m)
            
        # Show user on map
        LocateControl(auto_start=True).add_to(m)
        iframe = m.get_root()._repr_html_()
        return render_template(
            "cs_vis.html", iframe=iframe
        )
    return app
    
# Task examples
#@app.get("/add")
#def start_add() -> dict[str, object]:
#    a = 1
#    b = 2
#    result = tasks.add_together.delay(a, b)
#    return {"result_id": result.id}
    
#@app.get("/result/<id>")
#def task_result(id: str) -> dict[str, object]:
#    result = AsyncResult(id)
#    return {
#        "ready": result.ready(),
#        "successful": result.successful(),
#        "value": result.result if result.ready() else None,
#    }
    
