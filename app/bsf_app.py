import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_folium import st_folium
import folium
import geopandas as gpd
from folium.plugins import HeatMap
from folium import plugins
from time import sleep
#from google.cloud import storage

#from bsf_package.bsf_logic.design_streamlit import set_page_container_style
#from bsf_package.bsf_logic.maps import search_venue, heatmap_venues, display_district
#from bsf_package.bsf_logic.filterdata import filtercategory
#from bsf_package.bsf_logic.plots import plot_rating_berlin, plot_price_berlin,  plot_hist, plot_count_district

dir_name = os.path.abspath(os.path.dirname(__file__))
location = os.path.join(dir_name, 'clean_b1_veryfinal_categories.csv')
location2 = os.path.join(dir_name, 'neighbourhoods.geojson')

#import data and geojson
data = pd.read_csv(location)
geo_neighbourhoods = gpd.read_file(location2)

BACKGROUND_COLOR = 'pink'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 0, padding_right: int = 0, padding_left: int = 0, padding_bottom: int = 0,
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


def search_venue(df):
    return list(zip(df['lat'], df['lon']))

def heatmap_venues(data):
    latitude = 52.532538
    longitude = 13.520973
    # create map and display it
    map  = folium.Map(location=[latitude, longitude], zoom_start=12)
    boroughs_style = lambda x: {'color': 'black', 'opacity': 0.9, 'fillColor': 'green', 'weight': 0.6}
    folium.GeoJson(
      geo_neighbourhoods.geometry,
      style_function=boroughs_style,
      name='geojson'
      ).add_to(map)
    HeatMap(data).add_to(map)
    return map

# pin map
def display_district(data, neighbourhood_var):
    if neighbourhood_var == 'Berlin':
        district_df = data
    else:
        district_df = data[data.neighbourhood_group == neighbourhood_var]
        # Create an initial map of Berlin
        # Berlin latitude and longitude values
        latitude = 52.532538
        longitude = 13.520973
        # create map and display it
        berlin_map_district = folium.Map(location=[latitude, longitude], zoom_start=12)
    for i in range(0,len(district_df)):
        html=f"""
            <h4>{district_df.iloc[i]['title']}:</h4>
            <li>Rating: {district_df.iloc[i]['our_rating']}</li>
            <li>Nr. reviews: {district_df.iloc[i]['star_nr']}</li>
            <li>Categories: {district_df.iloc[i]['final_categories'][1:-1]}</li>
            """
        iframe = folium.IFrame(html=html, width=200, height=100)
        popup = folium.Popup(iframe, max_width=350)
        folium.Marker(
        location = [district_df.iloc[i]['lat'], district_df.iloc[i]['lon']],
        popup=popup, tooltip='Click for more information').add_to(berlin_map_district)
    return berlin_map_district

cat = ['Baby clothing store', 'Bag store','Beauty supplies store','Bridal store',"Children's clothing store",
 'Costume store','Department store','Tailor store','Fashion accessories store','Footwear store','Formal wear store',
 'General clothing store','Hat store','Home supplies store','Jeans store','Jewelry store','Leather store','Maternity store',
 "Men's clothing store",'Optical store','Outlet store','Pet store','Plus size clothing store','Second hand clothing store',
 'Shopping mall','Sportswear store','Swimwear store','T-shirt store','Underwear store','Vintage clothing store',
 'Wholesalers store',"Women's clothing store",'Work clothing store','Youth clothing store']

# WE DON'T NEED THIS ACTUALLY Calculate the amount of store categories per district
def shops_per_district(data, district):
    data = data[data["neighbourhood_group"] == district]
    data["categories_list"] = data.final_categories.apply(lambda x: x[1:-1].split(','))
    data.reset_index(inplace=True)
    data.drop(columns=['index'])
    all_categories = []
    for i in range(len(data)):
        for x in range(len(data.categories_list[i])):
            all_categories.append([data.categories_list[i][x]])
    for i in range(len(all_categories)):
        all_categories[i][0].strip()
        all_categories[i][0] = all_categories[i][0].strip()
    shop_count = {}
    for i in range(len(all_categories)):
        for shop in all_categories[i]:
            if shop in shop_count:
                shop_count[shop] = shop_count[shop] + 1
            else:
                shop_count[shop] = 1

    # TURN THE DICTIONARY INTO A DATAFRAME BEFORE PLOTTING IT
    # Turn the shop_count dictionary into a dataframe
    df = pd.DataFrame.from_dict(shop_count, orient='index')
    # Turn the store category into a column and reset index
    df = df.reset_index(names=['Store Category'])
    # Change the name for the column of Number of stores
    df.rename(columns = {0:'Number of stores'}, inplace = True)
    # Replace unnecessary characters from the dataframe
    df['Store Category'] = df['Store Category'].apply(lambda x: str(x).replace("'", ""))
    df['Store Category'] = df['Store Category'].apply(lambda x: str(x).replace('"', ""))
    # If you want the df in aphabetical order uncomment the next line
    # df.sort_values('Store Category', ascending=True)
    # If you want the df in ascending order (based on Number of stores) uncomment the next line
    # df.sort_values('Number of stores', ascending = True)
    # If you want the df in descending order (based on Number of stores) uncomment the next line
    # df.sort_values('Number of stores', ascending = False)

    # NOW PLOT THE DATAFRAME
    store_category = df['Store Category']
    num_stores = df['Number of stores']
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    # Horizontal Bar Plot
    ax.barh(store_category, num_stores)
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.',
            linewidth = 0.5,
            alpha = 0.2)
    # Show top values
    #ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.2,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')

    # Add Plot Title
    ax.set_title(f"Number of shops  in district",
                 loc ='center' )

    # Show Plot
    plt.show()
    return st.pyplot(plt)

def mean_per_district(data, district):

    data = data[data["neighbourhood_group"] == district]
    data["categories_list"] = data.final_categories.apply(lambda x: x[1:-1].split(','))
    data.reset_index(inplace=True)
    data.drop(columns=['index'])
    all_categories = []
    for i in range(len(data)):
        for x in range(len(data.categories_list[i])):
            all_categories.append([data.categories_list[i][x]])
    for i in range(len(all_categories)):
        all_categories[i][0].strip()
        all_categories[i][0] = all_categories[i][0].strip()
    shop_count = {}
    for i in range(len(all_categories)):
        for shop in all_categories[i]:
            if shop in shop_count:
                shop_count[shop] = shop_count[shop] + 1
            else:
                shop_count[shop] = 1
    mean_per_category = []
    for i in range(len(cat)):
      mask = data.final_categories.apply(lambda x: any(item for item in cat[i] if item in x))
      df2 = data[mask]
      tmp = df2['our_rating']
      mean_per_category.append(np.mean(tmp[i]))

    rat = pd.DataFrame({'Store Category': cat,'Mean rating': mean_per_category})
    # TURN THE DICTIONARY INTO A DATAFRAME BEFORE PLOTTING IT
    # Turn the shop_count dictionary into a dataframe
    df = pd.DataFrame.from_dict(shop_count, orient='index')
    # Turn the store category into a column and reset index

    df = df.reset_index()
    df = df.rename(columns = {'index': 'Store Category', 0:'Number of stores'})
    df['Store Category'] = df['Store Category'].apply(lambda x: str(x).replace("'", ""))
    df['Store Category'] = df['Store Category'].apply(lambda x: str(x).replace('"', ""))
    df = df.merge(rat, on = 'Store Category')
    df = df.sort_values('Mean rating', ascending = True)


    store_category = df['Store Category']
    num_stores = df['Number of stores']
    mean_stores = df['Mean rating']

    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    # Horizontal Bar Plot
    sns.set(font_scale=2)
    sns.set_theme(style="white",font="sans-serif", palette="Set2", rc={"font.size":20,"axes.titlesize":18})
    sns.barplot(y = 'Store Category', x = 'Mean rating', data = df, ci=False, orient = 'h').set(title=f'Mean Google rating of each shop type in {choice_district} in 2019-2022',xlabel ="", ylabel = "")
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.',
            linewidth = 0.5,
            alpha = 0.2)
    # Show top values
    #ax.invert_yaxis()

    # Add annotation to bars
    for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + 0.07, p.get_y()+p.get_height()/1.3, df['Number of stores'].loc[i],ha="center", fontsize = 12)
    plt.xlim(0, 5)
    plt.xlabel('Stars on Google Maps platform', fontsize=16)

    plt.grid(False)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # Show Plot
    return st.pyplot(plt)
# filter shop within list

def filtercategory(data, choice_shop):
    if choice_shop == 'All shops':
        df = df
    else:
        mask = data.final_categories.apply(lambda x: any(item for item in \
            choice_shop if item in x)) # filter df if any category matching
        df = data[mask]
    return df


### HERE THE APP BEGINS ###
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
choice_district = st.sidebar.selectbox('Choose a district',  ('Berlin',  'Charlottenburg-Wilm.', 'Friedrichshain-Kreuzberg', 'Lichtenberg', 'Marzahn - Hellersdorf', 'Mitte', 'Neukölln', 'Pankow',
'Reinickendorf', 'Spandau', 'Steglitz - Zehlendorf', 'Tempelhof - Schöneberg', 'Treptow - Köpenick'))

if choice_district == 'Berlin':
    choice_shop = st.sidebar.selectbox('Choose a shop type', ('Baby clothing store', 'Bag store','Beauty supplies store','Bridal store',"Children's clothing store",
 'Costume store','Department store','Tailor store','Fashion accessories store','Footwear store','Formal wear store',
 'General clothing store','Hat store','Home supplies store','Jeans store','Jewelry store','Leather store','Maternity store',
 "Men's clothing store",'Optical store','Outlet store','Pet store','Plus size clothing store','Second hand clothing store',
 'Shopping mall','Sportswear store','Swimwear store','T-shirt store','Underwear store','Vintage clothing store',
 'Wholesalers store',"Women's clothing store",'Work clothing store','Youth clothing store'))
else:
    choice_shop = st.sidebar.selectbox('Choose a shop type', ('Baby clothing store', 'Bag store','Beauty supplies store','Bridal store',"Children's clothing store",
 'Costume store','Department store','Tailor store','Fashion accessories store','Footwear store','Formal wear store',
 'General clothing store','Hat store','Home supplies store','Jeans store','Jewelry store','Leather store','Maternity store',
 "Men's clothing store",'Optical store','Outlet store','Pet store','Plus size clothing store','Second hand clothing store',
 'Shopping mall','Sportswear store','Swimwear store','T-shirt store','Underwear store','Vintage clothing store',
 'Wholesalers store',"Women's clothing store",'Work clothing store','Youth clothing store', 'All shops'))

if st.sidebar.button("Show results"):
    st.session_state.button_on = True
    st.session_state.gap_on = False

st.sidebar.header('Calculate')
choice_shop2 = st.sidebar.selectbox('Shop type', ('Baby clothing', 'Bag shop','Beauty supplies','Bridal store', "Children's clothing",'Costume store','Department store', 'Tailor store','Fashion accessories','Footwear','Formal wear','General clothing store','Hat shop','Home supplies','Jeans shop', 'Jewelry store', 'Leather store','Maternity store',"Men's clothing",'Optical store','Outlet store','Pet store','Plus size clothing','Second hand clothing','Shopping mall','Sportswear','Swimwear','T-shirt shop','Underwear','Vintage clothing store','Wholesalers',"Women's clothing",'Work clothing','Youth clothing'))

if st.sidebar.button('Show gap analysis'):
    st.session_state.gap_on = True
    st.session_state.button_on = False

if st.sidebar.button("Back to Home"):
    st.session_state.button_on = False
    st.session_state.gap_on = False

placeholder = st.empty()

# Main Page
with placeholder.container():
    latitude = 52.532538
    longitude = 13.520973
    # create map and display it
    m = folium.Map(location=[latitude, longitude], zoom_start=11.05)
    st_folium(m, width=1500, height=400)


if st.session_state.button_on:
    empty()
    if choice_district == 'Berlin':
        choice_shop = [choice_shop] #need it in list format for function filtercategory
        df = filtercategory(data, choice_shop)
        choice_shop = choice_shop[0] #take only string
        amount_shops = len(df) # amount of shops of that type in the selected district and category
        #header = f'{amount_shops} establishments are classified as {choice_shop.capitalize()} in Berlin.'
        #title = f'<p style="font-family:sans-serif; color:Black; font-size: 30px;"><b>{header}<b></p>'
        #st.markdown(title, unsafe_allow_html=True)
        st.info(f'{amount_shops} establishments are classified as {choice_shop} in Berlin.')

        if st.checkbox('Change to pinmap'):
            pin = display_district(df, choice_district)
            st_folium(pin,width=1400, height=400)
        else:
            dist = search_venue(df)
            heat = heatmap_venues(dist)
            st_folium(heat,width=1400, height=400)

        red_1 = df[['neighbourhood_group','our_rating']].groupby('neighbourhood_group', as_index = False).mean()
        red_2 = df[['neighbourhood_group','our_rating']].groupby('neighbourhood_group', as_index = False).count()
        red_2.rename(columns= {'our_rating':'count'}, inplace=True)
        red = red_1.merge(red_2, on = 'neighbourhood_group', how = "left")
        fig, ax= plt.subplots(figsize=(10, 5))
        sns.set(font_scale=2)
        sns.set_theme(style="white",font="sans-serif", palette="Set2", rc={"font.size":20,"axes.titlesize":16})
        sns.barplot(y = 'neighbourhood_group', x = 'our_rating', data = red, ci=False, orient = 'h').set(title=f'Mean Google rating of {choice_shop}s \nin each district in 2019-2022',xlabel ="", ylabel = "")
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + 0.07, p.get_y()+p.get_height()/1.4, red["count"].loc[i],ha="center", fontsize = 12)
        plt.xlim(0, 5)
        plt.xlabel('Stars on Google Maps platform', fontsize=14)
        plt.grid(False)
        plt.tick_params(axis='both', which='major', labelsize=14)
        st.pyplot(fig)

        #with col4:
         #plot the price (#### dont plot if 0!!!)
        df['€'] = df.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
        df['€€'] = df.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
        df['€€€'] = df.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
        x_price = df.neighbourhood_group.unique().tolist()
        y_price_cheap = df.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
        y_price_med = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
        y_price_exp = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()
        x_axis = np.arange(len(x_price))
        if sum(y_price_cheap) == 0 and sum(y_price_med) == 0 and sum(y_price_exp) == 0:
                    st.warning(f'Unfortunately, we do not have any data on price levels for {choice_shop}s.')
        else:
            fig2 = plt.figure(figsize=(10, 5))
            sns.set(font_scale = 2)
            plt.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
            plt.barh(x_axis, y_price_med,label="€€", height = 0.3)
            plt.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3    )
            plt.legend(fontsize = 10)
            plt.yticks(x_axis,x_price, fontsize = 14)
            plt.xticks(fontsize = 14)
            plt.xlabel('Nr. of shops', fontsize=14)
            plt.title(f'Price level of {choice_shop}s in each district**', fontsize=16)
            st.pyplot(fig2)


            mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''
            st.markdown(mystyle, unsafe_allow_html=True)

            c1, c2, c3 = st.columns((1, 1, 1))

            with c3:
                message2 = f'**Price information is based on a subset of shops with public price level on the Google Maps platform.'
                st.info(message2)


    else: # if choose a district


        if choice_shop == 'All shops':
            df = data[data["neighbourhood_group"] == choice_district]
            choice_shop = [choice_shop] #need it in list format for function filtercategory
            choice_shop = choice_shop[0] #take only string
            amount_shops = len(df) # amount of shops of that type in the selected district and category
            #col1, col2 = st.columns([10,1])
            #with col1:
            #st.header(f'{amount_shops} establishments categorize as clothing shops in {choice_district}') # uppercase the shop type
            st.info(f'{amount_shops} establishments categorize as clothing shops in {choice_district}.')

            if st.checkbox('Change to pinmap'):
                pin = display_district(df, choice_district)
                st_folium(pin,width=1400, height=400)
            else:
                dist = search_venue(df)
                heat = heatmap_venues(dist)
                st_folium(heat,width=1400, height=400)

            mean_per_district(df, choice_district)

        if choice_shop != 'All shops':
            df = data[data["neighbourhood_group"] == choice_district]
            choice_shop = [choice_shop] #need it in list format for function filtercategory
            df = filtercategory(df, choice_shop)
            choice_shop = choice_shop[0] #take only string
            amount_shops = len(df) # amount of shops of that type in the selected district and category

            if amount_shops > 0:
                #col1, col2 = st.columns([6,2]) # here adjust width of columns
                #with col1:
                #header = f'{amount_shops} establishments are classified as {choice_shop.capitalize()} in {choice_district}.'
                #title = f'<p style="font-family:sans-serif; color:Black; font-size: 30px;"><b>{header}<b></p>'
                #st.markdown(title, unsafe_allow_html=True)
                if amount_shops > 1:
                    st.info(f'{amount_shops} establishments are classified as {choice_shop} in {choice_district}.')
                elif amount_shops == 1:
                    st.info(f'{amount_shops} establishment is classified as {choice_shop} in {choice_district}.')


                if st.checkbox('Change to pinmap'):
                    pin = display_district(df, choice_district)
                    st_folium(pin,width=1400, height=400)
                else:
                    dist = search_venue(df)
                    heat = heatmap_venues(dist)
                    st_folium(heat,width=1400, height=400)


                #with col2:
                mean_rat = np.round(df['our_rating'].mean(),2)

                #message1 = f'- The mean rating for {choice_shop.capitalize()}s in {choice_district} is of {mean_rat} stars.'
                #title1 = f'<p style="font-family:sans-serif; color:Black; font-size: 20px;">{message1}</p>'
                #st.markdown(title1, unsafe_allow_html=True)
                st.success(f'- The mean Google maps rating from 2019 to 2022 for {choice_shop}s in {choice_district} is of {mean_rat} stars.')

                # price info
                cheap_shops = len(df[df['price'] == '€'])
                med_shops = len(df[df['price'] == '€€'])
                exp_shops = len(df[df['price'] == '€€€'])
                rest = df['price'].isna().sum()

                #plot_hist(df, choice_shop, choice_district)
                #col1, col2 = st.columns([5,5]) # here adjust width of columns
                col1, col2, col3 = st.columns([1,4,1])

                with col1:
                    st.write(' ')
                with col2:
                    fig = plt.figure(figsize=(15, 10))
                    sns.set(font_scale = 2)
                    sns.set_style("whitegrid")
                    #sns.set_theme(style="white",font="sans-serif", palette="Set2", rc={"font.size":20,"axes.titlesize":30})
                    sns.distplot(df['our_rating'], bins = 5, kde=False, hist_kws={'range':(0,5)}, color = 'blue')
                    fig.suptitle(f'Mean rating of establishments \nclassified as {choice_shop}s in {choice_district}', fontsize=24, fontdict={"weight": "bold"})
                    plt.xlabel('Mean Google rating 2019-2022', fontsize=22)
                    plt.ylabel(f'Amount of {choice_shop}s', fontsize=22)
                    st.pyplot(fig)
                with col3:
                    st.write(' ')

                if cheap_shops == 0 and med_shops == 0 and exp_shops == 0:
                    message2 = f'- There is no available price information for {choice_shop}s in this district.'
                else:
                    message2 = f'- There are {cheap_shops} low-price shops, {med_shops} medium-price shops and {exp_shops} high-price shops in the district. (Information about price is not available for {rest} shops.)'
                st.warning(message2)

            else:
                st.info(f'There are no establishments categorized as {choice_shop} in {choice_district}.')


if st.session_state.gap_on:
    empty()
    st.markdown("Gap analysis in here")
