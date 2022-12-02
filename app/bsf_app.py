import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap
from time import sleep
import os

from bsf_package.bsf_logic.heatmap import search_venue

dir_name = os.path.abspath(os.path.dirname(__file__))
# join the bobrza1.csv to directory to get file path
location = os.path.join(dir_name, 'geoshops.csv')
# join the route.csv to directory to get file path
location2 = os.path.join(dir_name, 'neighbourhoods.geojson')

#import data and geojson
#data = pd.read_csv('../data/geoshops.csv')
#geo_neighbourhoods = gpd.read_file("../data/neighbourhoods.geojson")
data = pd.read_csv(location)
geo_neighbourhoods = gpd.read_file(location2)


# Initialize session state for the button
if 'button_on' not in st.session_state:
    st.session_state.button_on = False
if 'gap_on' not in st.session_state:
    st.session_state.gap_on = False
# function for placeholder
def empty():
    placeholder.empty()
    sleep(0.01)

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


#Home page:
st.sidebar.markdown('Explore shop types in Berlin')
#Right side:  Two scrolling inputs
choice_district = st.sidebar.selectbox('Choose a district',  ('Berlin', 'Steglitz - Zehlendorf', 'Mitte', 'Friedrichshain-Kreuzberg',
       'Pankow', 'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg',
       'Neukölln', 'Reinickendorf', 'Spandau', 'Marzahn - Hellersdorf',
       'Treptow - Köpenick', 'Lichtenberg')) # District: list of 13 including Berlin
choice_shop = st.sidebar.selectbox('Choose a shop type', ('baby store', 'hat shop')) # Shop type: list of categories to be decided
if st.sidebar.button("Show results"):
    st.session_state.button_on = True
    st.session_state.gap_on = False

st.sidebar.markdown('Calculate gap analysis')
choice_shop = st.sidebar.selectbox('Choose a shop type', ("baby store", "women's clothing")) # Shop type: list of categories to be decided
if st.sidebar.button('Show gap analysis'):
    st.session_state.gap_on = True
    st.session_state.button_on = False

if st.sidebar.button("Back to Home"):
    st.session_state.button_on = False
    st.session_state.gap_on = False

placeholder = st.empty()
# Main Page
with placeholder.container():
    m= folium.Map(location=[52.5200, 13.405], zoom_start=5) # show map if no button pressed
    st_folium(m)


if st.session_state.button_on:
    empty()
    if choice_district == 'Berlin':
        st.markdown('Here the plots for the search')
        # df = data
        #st.markdown('There are {s} shops in your search')
    else:
        df1 = data[data["neighbourhood_group"] == choice_district]
        #st.markdown(len(df1))
        dist = search_venue(df1)
        heat = heatmap_venues(dist)
        st_folium(heat)

if st.session_state.gap_on:
    empty()
    st.markdown("Gap analysis in here")
