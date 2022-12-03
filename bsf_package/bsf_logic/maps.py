import os
import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap

dir_name = os.path.abspath(os.path.dirname(__file__))
location2 = os.path.join(dir_name, 'neighbourhoods.geojson')
geo_neighbourhoods = gpd.read_file('neighbourhoods.geojson')

def search_venue(df):
    return list(zip(df['lat'], df['lon']))

def heatmap_venues(data):
    map = folium.Map(location=[52.5200, 13.4050], zoom_start=11)
    boroughs_style = lambda x: {'color': 'black', 'opacity': 0.9, 'fillColor': 'green', 'weight': 0.6}
    folium.GeoJson(
      geo_neighbourhoods.geometry,
      style_function=boroughs_style,
      name='geojson'
      ).add_to(map)
    HeatMap(data).add_to(map)
    return map
