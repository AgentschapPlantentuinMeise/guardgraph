import os
from celery import Celery, Task, shared_task
from flask import Flask
from pygbif import species, occurrences
from guardgraph.graph import InteractionsGraph, EcoAnalysis

def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

#@shared_task(ignore_result=False)
#def add_together(a: int, b: int) -> int:
#    return a + b

@shared_task(ignore_result=False)
def get_case_study_polygon(case_study_name: str, polygon_simplifier: int = 1000) -> str:
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
    return case_study_polygon.wkt

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

def prep_speciesKey_list(species_list: list[str], with_interactors: bool = True) -> list[str]:
    if with_interactors:
        interactors = query_interactions(species_list)
    else: interactors = []
    speciesKeyList = set([
        (species.name_suggest(
            s, limit=1
        ) or [{'speciesKey':None}]
         )[0].get('speciesKey') for s in species_list
    ]+[
        (species.name_suggest(
            i['m']['name'], limit=1
        ) or [{'speciesKey':None}]
         )[0].get('speciesKey') # Only works for species
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

@shared_task(ignore_result=False)
def case_study_cube(
        case_study_name: str, speciesList: list[str],
        polygon_simplifier: int = 1000,
        with_interactors: bool = True) -> str:
    from shapely import wkt
    from guardgraph.gbif import cube_query
    case_study_polygon = wkt.loads(get_case_study_polygon(
        case_study_name, polygon_simplifier
    ))
    speciesKeyList = prep_speciesKey_list(speciesList, with_interactors)
    cube_job_id = cube_query(
        'christophe.vanneste@plantentuinmeise.be',
        'cvanneste',
        open('gbif_pwd','rt').read().strip(),
        case_study_polygon.wkt,
        speciesKeyList
    )
    return cube_job_id

@shared_task(ignore_result=False)
def case_study_interactions(
        case_study_name: str, speciesList: list[str],
        polygon_simplifier: int = 1000,
        interaction_cube_id: str|None = None) -> dict[str, object]:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    cs_dir = f'/mbg/instance/shared/{case_study_name}'
    if not os.path.exists(cs_dir):
        os.mkdir(cs_dir)
    ig = InteractionsGraph()
    interactions = pd.DataFrame({'species': speciesList})
    interactions['ix_r_types'] = interactions.species.apply(

        lambda x: ig.run_query(

            f"MATCH (n)-[r]-() WHERE n.name = '{x}' RETURN TYPE(r), COUNT(*)"
        )
    )
    rtypes = {r:0 for r in ig.relationships}
    r_types = pd.DataFrame(
        list(interactions.ix_r_types.apply(
            lambda x: rtypes|{r['TYPE(r)']:r['COUNT(*)'] for r in x})), index=interactions.index)
    r_types['total_r'] = r_types.sum(axis=1)
    r_types['species_name'] = interactions.species
    top20ix = r_types.set_index(
        'species_name'
    ).sum().sort_values(ascending=False)
    top20ix = top20ix[top20ix>0].index[1:21]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(
        data=r_types.set_index('species_name')[top20ix],
        orient='h', ax=ax
    )
    ax.set_xlabel('Interactions / species')
    fig.tight_layout()
    fig.savefig(f'{cs_dir}/ix_types_kaggle.svg')

    output_files = {
        'All interactions involving species of interest': f'/mbg/instance/shared/{case_study_name}/ix_types_kaggle.svg'
    }
    
    species_with_ix = list(
        r_types[r_types.total_r>0].species_name
    )

    ## Dyadic
    dyadic_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list       
       WITH n MATCH (n)-[r]-(m) WHERE m.name IN $species_list
       RETURN TYPE(r), COUNT(*)
''',
                            species_list=species_with_ix
    )
    dyadic_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list       
       WITH n MATCH (n)-[r]-(m) WHERE m.name IN $species_list
       AND m.name <> n.name
       RETURN n.name,TYPE(r),m.name
''',
                            species_list=species_with_ix)
    dyadic_nodes = pd.DataFrame(dyadic_nodes)

    ## Triadic
    triadic_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o) WHERE m.name IN $species_list         
       AND o.name in $species_list RETURN TYPE(r),TYPE(q),COUNT(*)            
''',
                             species_list=species_with_ix)
    triadic_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o) WHERE m.name IN $species_list         
       AND o.name in $species_list
       RETURN n.name,TYPE(r),m.name,TYPE(q),o.name
''',
                                 species_list=species_with_ix)
    triadic_nodes = pd.DataFrame(triadic_nodes)
    triadic_nodes['TYPESTR'] = triadic_nodes.apply(
        lambda x: str(
            pd.Series([x['TYPE(r)'], x['TYPE(q)']])
            .value_counts().sort_index().to_dict()
        ), axis=1
    )
    
    triadic_closed_r = ig.run_query('''MATCH (n) WHERE n.name IN $species_list                                                                      
       WITH n MATCH (n)-[r]-(m)-[q]-(o)-[s]-(n) WHERE m.name IN $species_list  
       AND o.name in $species_list RETURN TYPE(r),TYPE(q),TYPE(s),COUNT(*)     
''',
                                    species_list=species_with_ix)
    triadic_closed_nodes = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
       WITH n MATCH (n)-[r]-(m)-[q]-(o)-[s]-(n) WHERE m.name IN $species_list  
       AND o.name in $species_list
       RETURN n.name,TYPE(r),m.name,TYPE(q),o.name,TYPE(s)
''',
                                    species_list=species_with_ix) 
    triadic_closed_nodes = pd.DataFrame(triadic_closed_nodes)
    triadic_closed_nodes['TYPESTR'] = triadic_closed_nodes.apply(
        lambda x: str(
            pd.Series([x['TYPE(r)'], x['TYPE(q)'], x['TYPE(s)']])
            .value_counts().sort_index().to_dict()
        ), axis=1
    )
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=((5,15)))
    dyadic_nodes['TYPE(r)'].value_counts().plot.barh(ax=axes[0])
    axes[0].set_ylabel('Dyadic relationships')
    (triadic_nodes['TYPESTR'].value_counts()/2).plot.barh(ax=axes[1])
    axes[1].set_ylabel('Open triads')
    (triadic_closed_nodes['TYPESTR'].value_counts()/3).plot.barh(ax=axes[2])
    axes[2].set_ylabel('Closed triads')
    fig.tight_layout()
    fig.savefig(f'{cs_dir}/internal_network_structure.svg')
    output_files['Internal network structure'] = f'{cs_dir}/internal_network_structure.svg'
    
    species_kingdom = ig.run_query('''MATCH (n) WHERE n.name IN $species_list
RETURN n.kingdom,COUNT(*)''', species_list=species_with_ix)
    species_kingdom = pd.DataFrame(species_kingdom).rename(
        {'n.kingdom':'Kingdom', 'COUNT(*)': 'Count'}, axis=1
    )
    output_files['Species kingdom distribution'] = f'{cs_dir}/species_kingdom.csv'
    species_kingdom.to_csv(output_files['Species kingdom distribution'], index=False)
    
    ea = EcoAnalysis(ig)
    r_types_count = r_types.drop(
        ['species_name','total_r'],axis=1
    ).sum()
    ea.create_projection(
        'guardin_projection',
        ['species'],
        {
            r:{'properties':['occurrences']}
            for r in r_types_count[r_types_count>0].index
        },
        force=True
    )
    ea.create_embedding(
        'guardin_projection',
        embeddingDimension=10, force=True
    )
    embeddings = ea.get_embeddings(
        'guardin_projection',
        species_list=species_with_ix, plot=False
    )
    ea.drop_projection('guardin_projection')
    embeddings = embeddings.groupby('name').apply(
        lambda g:  np.mean(np.stack(g.embedding), axis=0)
    )
    embeddings = pd.DataFrame(
        embeddings.to_list(), index=embeddings.index
    )
    output_files['Species embeddings'] = f'{cs_dir}/embeddings.csv'
    embeddings.to_csv(output_files['Species embeddings'])

    #if interaction_cube_id:
        #TODO
    
    return {
        'title': f'{case_study_name} interactions analysis',
        'files': output_files
    }

@shared_task(ignore_result=False)
def analyze_casestudy_data(gbif_cube):
    import re
    import zipfile
    import pandas as pd
    from shapely import Point
    import geopandas as gpd
    from intercubos.gridit import Grid
    import folium
    from folium import plugins
    # Prep data
    zip = zipfile.ZipFile(gbif_cube)                      
    zipfile = zip.open(zip.infolist()[0])
    cube = pd.read_table(zipfile)
    eeacellcode = re.compile(
        r'(?P<grid_size>\d)+kmE(?P<longitude>\d+)N(?P<latitude>\d+)'
    )
    cube = pd.concat(
        (
            cube,
            cube.eeacellcode.apply(
                lambda x:  pd.Series(eeacellcode.match(x).groupdict()))
        ), axis=1
    )
    cube.latitude = cube.latitude.astype(int)*1000 # *1000 because 1000m grid cell size
    cube.longitude = cube.longitude.astype(int)*1000
    cube = gpd.GeoDataFrame(
        cube,
        geometry=cube.apply(
            lambda x: Point(x.latitude,x.longitude),axis=1
        ), crs=3035#4326
    ).to_crs(epsg=4326)
    grid = Grid(
        *cube.total_bounds,
        # lat and lon still in 3035 coords
        #cube.latitude.min(), cube.longitude.min(),
        #cube.latitude.max(), cube.longitude.max(),
        stepsize=1000
    )
    grid.assign_to_grid(cube)
    grid.remove_empty_grid_cells()
    heat_data = [[point.xy[1][0], point.xy[0][0]] for point in cube.geometry]

    # prep map
    m = folium.Map(
        location=(cube.latitude.mean(),cube.longitude.mean()),
        prefer_canvas=True, zoom_start=12
    )
    plugins.HeatMap(heat_data).add_to(m)
    iframe = m.get_root()._repr_html_()
    return iframe

