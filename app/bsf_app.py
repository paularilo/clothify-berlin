import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap


# heatmaps for shop distribution in berlin
#def search_venue(df, category):
 #   search = lambda x:True if category.capitalize() in x else False
  #  venues = df[df['categoryName'].apply(search)].reset_index(drop='index')
   # venues_lat_long = list(zip(venues['lat'], venues['lon']))
    #return venues_lat_long

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
btn = st.sidebar.button("Show results")

st.sidebar.markdown('Calculate gap analysis')
choice_shop = st.sidebar.selectbox('Choose a shop type', ("baby store", "women's clothing")) # Shop type: list of categories to be decided
gap = st.sidebar.button('Show gap analysis')

chckbox = st.sidebar.button("Back to Home")

placeholder = st.empty()

if not btn and not gap:
    m= folium.Map(location=[52.5200, 13.405], zoom_start=5) # show map if no button pressed
    st_data = st_folium(m)

elif btn: # if show results button pressed
    data = pd.read_csv('data/geoshops.csv')
    geo_neighbourhoods = gpd.read_file("data/neighbourhoods.geojson")

    if choice_district == 'Berlin':

        st.markdown('Here the plots for the search')
        df = data


        #st.markdown('There are {s} shops in your search')
    else:
        df1 = data[data["neighbourhood_group"] == choice_district]
        #st.markdown(len(df1))
        #dist = search_venue(df1)
        #heat = heatmap_venues(dist)
        heat= folium.Map(location=[54.5200, 3.405], zoom_start=5) # show map if no button pressed
        st_heat = st_folium(heat)

        #st.markdown('Here the plots for the search')

        #st.markdown(len(df1))

    #data = dist[dist["categories"] == choice_district]

elif gap: # if gap analysis button pressed
    df = pd.DataFrame(range(0, 10))
    placeholder.dataframe(df)

if chckbox:
    btn = False
    #gap = False
