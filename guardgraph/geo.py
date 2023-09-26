import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from OSMPythonTools.overpass import Overpass
from OSMPythonTools.nominatim import Nominatim

def plot_countries(countries, ax=None):
    """
    Args:
      countries (list of tuples): Should be list of tuples `('Name', admin_level)`

    Example:
      >>> plot_countries[
      ... ('France métropolitaine', 3),
      ... ('England', 4),
      ... ('Cymru / Wales', 4),
      ... ('Alba / Scotland', 4)
      ... ])
    
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10,10))
    else: fig = ax.get_figure()
    ax.axis('equal')
        
    overpass = Overpass()
    #result = overpass.query('rel[admin_level=3]["name"="France métropolitaine"]; way(r); out body geom;')
    for name, level in countries:
        result = overpass.query(
            f'rel[admin_level={level}]["name"="{name}"]; out body geom;')
        r = result.relations()[0]
        g = r.geometry()
        for polygon in g['coordinates']:
            polygon = np.array(
                polygon if g['type'] == 'Polygon'
                else polygon[0] # Multipolygon
            )
            ax.fill(*polygon.T, 'whitesmoke')
            ax.plot(*polygon.T, 'gainsboro')
    return fig, ax
        
#polygon_data = pd.DataFrame({'elements':result.elements()})
#for t in ('border_type', 'boundary'): polygon_data[t] = polygon_data.elements.apply(lambda x: x.tag(t))
#polygon_data['coordinates'] = polygon_data.elements.apply(lambda x: np.array(x.geometry()['coordinates']))
#polygon_data = polygon_data.loc[polygon_data.coordinates.apply(lambda x: len(x.shape)!=2)].copy()



# import lxml
# lxml.__version__
# xml = '<a xmlns="test"><b xmlns="test"/></a>'
# from lxml import etree
# root = etree.fromstring(xml)
# root
# from bs4 import BeautifulSoup
# x=BeautifulSoup(xml, 'xml')
# print(x)
# from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
# result_rwn = overpass.query('rel[admin_level=3]["name"="France métropolitaine"]; (._; way(r); node(r);); out body geom meta;')                                       
# overpass = Overpass()
# result_rwn = overpass.query('rel[admin_level=3]["name"="France métropolitaine"]; (._; way(r); node(r);); out body geom meta;')                                       
# r=result_rwn.relations()[0]
# r.geometry()
