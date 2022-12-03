
def shopprice(df):

    berlin_price = berlin[['categoryName', 'lat','lon', 'price','price_cont', 'neighbourhood', 'neighbourhood_group']]
    berlin_price['€'] = berlin.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
    berlin_price['€€'] = berlin.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
    berlin_price['€€€'] = berlin.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
    x_price = berlin_price.neighbourhood_group.unique().tolist()
    y_price_cheap = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
    y_price_med = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
    y_price_exp = berlin_price.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()

    x_axis = np.arange(len(x_price))

    plt.figure(figsize=(20,15))

    plt.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
    plt.barh(x_axis, y_price_med,label="€€", height = 0.3)
    plt.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3)
    plt.yticks(x_axis,x_price)
    plt.legend()
    return plt.show()
