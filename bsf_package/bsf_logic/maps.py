import os
import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap
from IPython.display import display
from folium import plugins

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

# problem with loc
# pin map
def display_district(data, neighbourhood_var):
    if neighbourhood_var == 'Berlin':
        district_df = data
    else:
        district_df = data[data.neighbourhood_group == neighbourhood_var]
        # Create an initial map of Berlin
        # Berlin latitude and longitude values

    latitude = 52.520008
    longitude = 13.404954
    # create map and display it
    berlin_map_district = folium.Map(location=[latitude, longitude], zoom_start=12)
    for i in range(0,len(district_df)):
        folium.Marker(
        location=[district_df.loc[i]['lat'], district_df.loc[i]['lon']].add_to(berlin_map_district))
        #popup=district_df.iloc[i]['title'], tooltip='Click for more information').add_to(berlin_map_district)
    return berlin_map_district
