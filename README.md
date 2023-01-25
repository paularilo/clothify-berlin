# Clothify

## Project summary
The project aims to help potential enterpreneurs in the clothing retail industry to identify the perfect location to open their store in Berlin. As such, it is mostly focused on a business analysis with two main parts: an exploratory and a gap analysis function. 

### Exploratory function
Based on the user's neighborhood (or whole Berlin) and their choice of shop type, the product provides different textual and graphical information. Initially, it provides the amount of stores that sell that category in the chosen neighborhood, alongside with a heatmap that shows the distribution of them in the neighborhood or in the whole Berlin. The heatmap can be changed to a pin-map, which lists the above stores and provides information such as: name of the store, rating and other categories that this store sells. Additionally, the user is provided with 2 bar charts that show the mean rating (for the period 2019 - 2022) and the price level of all the shops of the chosen category per district. The amount of shops based on which  that mean rating was calculated is provided. When the user inputs only a certain neighborhood and all types of shops, the barchart provided alongside the already mentioned maps, shows the mean rating (for the period 2019 - 2022) of the choses category per district, and the amount of shops based on which this rating was calculated.

### Gap Analysis function
Based on the user's chosen shop category, the product provides a list with Berlin's 10 top neighbourhoods to open a store of the chosen category based on several criteria (see How did we do it section). The result is displayed in textual format and as coloured map.

## How did we do it

Our app is based on Python 3 and hosted in Streamlit. Data collection (shops ID, location, price, reviews) took place via Google API services. Information about Berlin demographics were collected from the data base of Berlin Open Data (https://daten.berlin.de/).

For the exploratory part, we used pandas and numpy to perform statistical analysis on the collected data and create graphical representations using seaborn and matplotlib. We used Scipy to combine demographic information about Berlin neighbourhoods and calculate the best 10 places to open a specific type of shop (e.g., baby clothing store) in Berlin based on rent price, amount of neighbouring shops of the same type and rating of neighbouring shops.

## Our product
The product is available in Streamlit: https://clothify-berlin.streamlit.app/

