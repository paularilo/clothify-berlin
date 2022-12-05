import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap
from time import sleep
import os
import matplotlib.pyplot as plt
import numpy as np
#from bsf_package.bsf_logic.heatmap import search_venue
#from common import set_page_container_style

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )


dir_name = os.path.abspath(os.path.dirname(__file__))
location = os.path.join(dir_name, 'clean_b1_veryfinal_categories.csv')
location2 = os.path.join(dir_name, 'neighbourhoods.geojson')

#import data and geojson
data = pd.read_csv(location)
geo_neighbourhoods = gpd.read_file(location2)

# page padding
st.set_page_config(
    page_title='Shopify',
    layout='wide',
    page_icon=':rocket:'
)

set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0
)


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
    map = folium.Map(location=[52.5200, 13.4050], zoom_start=10)
    boroughs_style = lambda x: {'color': 'black', 'opacity': 0.9, 'fillColor': 'green', 'weight': 0.6}
    folium.GeoJson(
      geo_neighbourhoods.geometry,
      style_function=boroughs_style,
      name='geojson'
      ).add_to(map)
    HeatMap(data).add_to(map)
    return map

st.title('Clothify')
#st.sidebar.header('Settings')


#Home page:
with st.sidebar:
    st.header('Explore shop types in Berlin')
#Right side:  Two scrolling inputs
    choice_district = st.selectbox('Choose a district',  ('Berlin', 'Steglitz - Zehlendorf', 'Mitte', 'Friedrichshain-Kreuzberg',
        'Pankow', 'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg',
        'Neukölln', 'Reinickendorf', 'Spandau', 'Marzahn - Hellersdorf',
        'Treptow - Köpenick', 'Lichtenberg')) # District: list of 13 including Berlin
    choice_shop = st.sidebar.selectbox('Choose a shop type', ('Baby clothing', 'Bag shop','Beauty supplies','Bridal store', "Children's clothing",'Costume store','Department store',
    'Emboidery & Clothing alternation','Fashion accessories','Footwear','Formal wear','General clothing store','Hat shop','Home supplies','Jeans shop', 'Jewelry store', 'Leather store','Maternity store',"Men's clothing",'Optical store','Outlet store','Pet store','Plus size clothing','Second hand clothing','Shopping mall','Sportswear','Swimwear','T-shirt shop','Underwear','Vintage clothing store','Wholesalers',"Women's clothing",'Work clothing','Youth clothing'))

    if st.button("Show results"):
        st.session_state.button_on = True
        st.session_state.gap_on = False

    st.header('Calculate gap analysis')
    choice_shop = st.sidebar.selectbox('Shop type', ('Baby clothing', 'Bag shop','Beauty supplies','Bridal store', "Children's clothing",'Costume store','Department store',
    'Emboidery & Clothing alternation','Fashion accessories','Footwear','Formal wear','General clothing store','Hat shop','Home supplies','Jeans shop', 'Jewelry store', 'Leather store','Maternity store',"Men's clothing",'Optical store','Outlet store','Pet store','Plus size clothing','Second hand clothing','Shopping mall','Sportswear','Swimwear','T-shirt shop','Underwear','Vintage clothing store','Wholesalers',"Women's clothing",'Work clothing','Youth clothing'))

    if st.button('Show gap analysis'):
        st.session_state.gap_on = True
        st.session_state.button_on = False

    if st.button("Back to Home"):
        st.session_state.button_on = False
        st.session_state.gap_on = False

    placeholder = st.empty()

# Main Page
with placeholder.container():
    m= folium.Map(location=[52.5200, 13.405], zoom_start=12) # show map if no button pressed
    st_folium(m, width=1500, height=600)

if st.session_state.button_on:
    empty()
    if choice_district == 'Berlin':
        col1, col2 = st.columns([6,5])
        choice_shop = [choice_shop]
        mask = data.final_categories.apply(lambda x: any(item for item in choice_shop if item in x))
        df = data[mask]
        choice_shop = choice_shop[0]

        #df1 = data[data["categoryName"] == choice_shop]
        amount_shops = len(df)

        with col1:
            st.markdown(f'There is a total of {amount_shops} {choice_shop.capitalize()}s shops in Berlin') # uppercase the shop type
            dist = search_venue(df)
            heat = heatmap_venues(dist)
            st_folium(heat,width=500, height=500)
            st.checkbox('Change to pinmap')

        with col2:
            #st.header(f"Mean rating of {choice_shop.capitalize()}s in each district")
            red = df[['neighbourhood_group','our_rating']].groupby('neighbourhood_group', as_index = False).mean()

            fig = plt.figure(figsize=(15, 15))
            #st.markdown(x)
            x = red['neighbourhood_group']
            y = red['our_rating']

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e','#e377c2']
            plt.xlim(0,5)
            plt.title(f'Mean rating of {choice_shop.capitalize()}s in each district', fontsize=35)
            plt.barh(x,y,color=colors)
            plt.yticks(fontsize = 30)
            plt.xticks(fontsize = 30)
            st.pyplot(fig)


            berlin_price = df[['categoryName', 'lat','lon', 'price','price_cont', 'neighbourhood', 'neighbourhood_group']]
            berlin_price['€'] = df.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
            berlin_price['€€'] = df.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
            berlin_price['€€€'] = df.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
            x_price = berlin_price.neighbourhood_group.unique().tolist()
            y_price_cheap = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
            y_price_med = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
            y_price_exp = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()

            x_axis = np.arange(len(x_price))

            fig2 = plt.figure(figsize=(15, 20))
            plt.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
            plt.barh(x_axis, y_price_med,label="€€", height = 0.3)
            plt.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3    )
            plt.legend(fontsize = 20)
            plt.yticks(x_axis,x_price, fontsize = 30)
            plt.xticks(fontsize = 30)
            plt.title(f'Price level of {choice_shop.capitalize()}s in each district', fontsize=35)
            st.pyplot(fig2)


    else:
        df = data[data["neighbourhood_group"] == choice_district]
        #st.markdown(len(df1))
        col1, col2 = st.columns([6,5])

        with col1:
            amount_shops = len(df1)
            st.header(f'There is a total of {amount_shops} shops in {choice_district}') # uppercase the shop type
            dist = search_venue(df1)
            heat = heatmap_venues(dist)
            st_folium(heat,width=500, height=500)
            st.checkbox('Change to pinmap')

        with col2:
            #st.header(f"Mean rating of {choice_shop.capitalize()}s in each district")
            red = df[['neighbourhood_group','our_rating']].groupby('neighbourhood_group', as_index = False).mean()

            fig = plt.figure(figsize=(15, 15))
            #st.markdown(x)
            x = red['neighbourhood_group']
            y = red['our_rating']

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e','#e377c2']
            plt.xlim(0,5)
            plt.title(f'Mean rating of {choice_shop.capitalize()}s in {choice_district}', fontsize=35)
            plt.barh(x,y,color=colors)
            plt.yticks(fontsize = 30)
            plt.xticks(fontsize = 30)
            st.pyplot(fig)


            berlin_price = df[['categoryName', 'lat','lon', 'price','price_cont', 'neighbourhood', 'neighbourhood_group']]
            berlin_price['€'] = df.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
            berlin_price['€€'] = df.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
            berlin_price['€€€'] = df.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
            x_price = berlin_price.neighbourhood_group.unique().tolist()
            y_price_cheap = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
            y_price_med = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
            y_price_exp = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()

            x_axis = np.arange(len(x_price))

            fig2 = plt.figure(figsize=(15, 20))
            plt.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
            plt.barh(x_axis, y_price_med,label="€€", height = 0.3)
            plt.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3    )
            plt.legend(fontsize = 20)
            plt.yticks(x_axis,x_price, fontsize = 30)
            plt.xticks(fontsize = 30)
            plt.title(f'Price level of {choice_shop.capitalize()}s in {choice_district}', fontsize=35)
            st.pyplot(fig2)



if st.session_state.gap_on:
    empty()
    st.markdown("Gap analysis in here")
