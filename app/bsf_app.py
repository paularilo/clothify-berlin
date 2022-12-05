import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
#from folium.plugins import HeatMap
from folium import plugins
from time import sleep

from bsf_package.bsf_logic.design_streamlit import set_page_container_style
from bsf_package.bsf_logic.maps import search_venue, heatmap_venues, display_district
from bsf_package.bsf_logic.filterdata import filtercategory
from bsf_package.bsf_logic.plots import plot_rating_berlin, plot_price_berlin,  plot_hist, plot_count_district

dir_name = os.path.abspath(os.path.dirname(__file__))
location = os.path.join(dir_name, 'clean_b1_veryfinal_categories.csv')
location2 = os.path.join(dir_name, 'neighbourhoods.geojson')

#import data and geojson
data = pd.read_csv(location)
geo_neighbourhoods = gpd.read_file(location2)

#first step streamlit
st.set_page_config(
    page_title='Shopify',
    layout='wide',
    page_icon=':rocket:'
)

# padding
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

st.title('Clothify')
#st.sidebar.header('Settings')

st.sidebar.header('Explore shop types in Berlin')
choice_district = st.sidebar.selectbox('Choose a district',  ('Berlin', 'Steglitz - Zehlendorf', 'Mitte', 'Friedrichshain-Kreuzberg',
       'Pankow', 'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg',
       'Neukölln', 'Reinickendorf', 'Spandau', 'Marzahn - Hellersdorf',
       'Treptow - Köpenick', 'Lichtenberg')) # District: list of 13 including Berlin
if choice_district == 'Berlin':
    choice_shop = st.sidebar.selectbox('Choose a shop type', ('Baby clothing store', 'Bag store','Beauty supplies store','Bridal store',"Children's clothing store",
 'Costume store','Department store','Emboidery & Clothing alternation store','Fashion accessories store','Footwear store','Formal wear store',
 'General clothing store','Hat store','Home supplies store','Jeans store','Jewelry store','Leather store','Maternity store',
 "Men's clothing store",'Optical store','Outlet store','Pet store','Plus size clothing store','Second hand clothing store',
 'Shopping mall','Sportswear store','Swimwear store','T-shirt store','Underwear store','Vintage clothing store',
 'Wholesalers store',"Women's clothing store",'Work clothing store','Youth clothing store'))
else:
    choice_shop = st.sidebar.selectbox('Choose a shop type', ('Baby clothing store', 'Bag store','Beauty supplies store','Bridal store',"Children's clothing store",
 'Costume store','Department store','Emboidery & Clothing alternation store','Fashion accessories store','Footwear store','Formal wear store',
 'General clothing store','Hat store','Home supplies store','Jeans store','Jewelry store','Leather store','Maternity store',
 "Men's clothing store",'Optical store','Outlet store','Pet store','Plus size clothing store','Second hand clothing store',
 'Shopping mall','Sportswear store','Swimwear store','T-shirt store','Underwear store','Vintage clothing store',
 'Wholesalers store',"Women's clothing store",'Work clothing store','Youth clothing store', 'All shops'))

if st.sidebar.button("Show results"):
    st.session_state.button_on = True
    st.session_state.gap_on = False

st.sidebar.header('Calculate')
choice_shop2 = st.sidebar.selectbox('Shop type', ('Baby clothing', 'Bag shop','Beauty supplies','Bridal store', "Children's clothing",'Costume store','Department store', 'Emboidery & Clothing alternation','Fashion accessories','Footwear','Formal wear','General clothing store','Hat shop','Home supplies','Jeans shop', 'Jewelry store', 'Leather store','Maternity store',"Men's clothing",'Optical store','Outlet store','Pet store','Plus size clothing','Second hand clothing','Shopping mall','Sportswear','Swimwear','T-shirt shop','Underwear','Vintage clothing store','Wholesalers',"Women's clothing",'Work clothing','Youth clothing'))

if st.sidebar.button('Show gap analysis'):
    st.session_state.gap_on = True
    st.session_state.button_on = False

if st.sidebar.button("Back to Home"):
    st.session_state.button_on = False
    st.session_state.gap_on = False

placeholder = st.empty()

# Main Page
with placeholder.container():
    m = folium.Map(location=[52.5200, 13.405], zoom_start=12) # show map if no button pressed
    st_folium(m, width=1500, height=600)

if st.session_state.button_on:
    empty()
    if choice_district == 'Berlin':
        choice_shop = [choice_shop] #need it in list format for function filtercategory
        df = filtercategory(data, choice_shop)
        choice_shop = choice_shop[0] #take only string
        col1, col2 = st.columns([7,2]) # here adjust width of columns

        with col1:
            #df = filtercategory(data, choice_shop)
            amount_shops = len(df) # amount of shops of that type in the selected district and category
            st.markdown(f'{amount_shops} establishments are classified as {choice_shop.capitalize()} in Berlin')
            dist = search_venue(df)
            heat = heatmap_venues(dist)
            st_folium(heat,width=1000, height=400)
            #pin = display_district(df, choice_district)
            #st_folium(pin,width=1000, height=400)
        with col2:
            st.checkbox('Change to pinmap')

        col3, col4 = st.columns([5,5]) # here adjust width of columns

        with col3:
            plot_rating_berlin(df, choice_shop)

        with col4:
            plot_price_berlin(df, choice_shop)

    else: # if choose a district
        df = data[data["neighbourhood_group"] == choice_district]

        if choice_shop == 'All shops':
            amount_shops = len(df)
            col1, col2 = st.columns([10,1])
            with col1:
                st.header(f'{amount_shops} establishments categorize as clothing shops in {choice_district}') # uppercase the shop type
                dist = search_venue(df)
                heat = heatmap_venues(dist)
                st_folium(heat,width=500, height=500)
                #plot_count_district(df)
            with col2:
                pass
                #st.checkbox('Change to pinmap')

        elif choice_shop != 'All shops':
            df = filtercategory(df, choice_shop)
            amount_shops = len(df)
            if amount_shops > 0:
                col1, col2 = st.columns([6,2]) # here adjust width of columns
                with col1:
                    st.header(f'{amount_shops} establishments are classified as {choice_shop.capitalize()} in {choice_district}') # uppercase the shop type
                    dist = search_venue(df)
                    heat = heatmap_venues(dist)
                    st_folium(heat,width=500, height=500)

                with col2:
                    mean_rat = np.round(df['our_rating'].mean(),2)
                    cheap_shops = len(df[df['price'] == '€'])
                    med_shops = len(df[df['price'] == '€€'])
                    exp_shops = len(df[df['price'] == '€€€'])
                    rest = df['price'].isna().sum()

                    st.markdown(f'The mean rating of {choice_shop.capitalize()}s in {choice_district} is of {mean_rat} stars.')
                    st.markdown(f'There are {cheap_shops} low-price shops, {med_shops} medium-price shops and {exp_shops} high-price shops in the district. (Price information is not yet available for {rest} shops.)')
                    plot_hist(df, choice_shop, choice_district)
            else:
                st.header(f'There are no establishments categorized as {choice_shop.capitalize()} in {choice_district}.')


if st.session_state.gap_on:
    empty()
    st.markdown("Gap analysis in here")
