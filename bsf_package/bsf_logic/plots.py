import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_rating_berlin(df, choice_shop):
    fig = plt.figure(figsize=(15, 20))
    #st.markdown(x)
    red = df[['neighbourhood_group','our_rating']].groupby('neighbourhood_group', as_index = False).mean()
    x = df['neighbourhood_group']
    y = df['our_rating']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e','#e377c2']
    plt.xlim(0,5)
    plt.title(f'Mean rating of {choice_shop.capitalize()}s in each district', fontsize=35)
    plt.barh(x,y,color=colors)
    plt.yticks(fontsize = 30)
    plt.xticks(fontsize = 30)
    return st.pyplot(fig)


def plot_price_berlin(df, choice_shop):
    df['€'] = df.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
    df['€€'] = df.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
    df['€€€'] = df.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
    x_price = df.neighbourhood_group.unique().tolist()
    y_price_cheap = df.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
    y_price_med = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
    y_price_exp = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()

    x_axis = np.arange(len(x_price))

    fig2 = plt.figure(figsize=(15, 20))
    plt.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
    plt.barh(x_axis, y_price_med,label="€€", height = 0.3)
    plt.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3    )
    plt.legend(fontsize = 20)
    plt.yticks(x_axis,x_price, fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.title(f'Price level of {choice_shop.capitalize()}s in each district', fontsize=35)
    return st.pyplot(fig2)

def subplots_berlin(df, choice_shop):
    fig = plt.figure()

    #rating subplot
    ax1 = fig.add_subplot(1, 1, 1)  # equivalent but more general
    x = df['neighbourhood_group']
    y = df['our_rating']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e','#e377c2']
    ax1.xlim(0,5)
    ax1.title(f'Mean rating of {choice_shop.capitalize()}s in each district', fontsize=35)
    ax1.barh(x,y,color=colors)
    ax1.yticks(fontsize = 30)
    ax1.xticks(fontsize = 30)

    #price subplot
    ax2 = fig.add_subplot(1, 2, 2)  # equivalent but more general
    df['€'] = df.price.map({"€": 1,"€€":0,"€€€":0}).fillna(0)
    df['€€'] = df.price.map({"€": 0,"€€":1,"€€€":0}).fillna(0)
    df['€€€'] = df.price.map({"€": 0,"€€":0,"€€€":1}).fillna(0)
    x_price = df.neighbourhood_group.unique().tolist()
    y_price_cheap = df.groupby(['neighbourhood_group'],as_index=False).sum()['€'].tolist()
    y_price_med = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€'].tolist()
    y_price_exp = df.groupby(['neighbourhood_group'],as_index=False).sum()['€€€'].tolist()
    x_axis = np.arange(len(x_price))
    ax2.barh(x_axis - 0.3, y_price_cheap, label="€", height = 0.3)
    ax2.barh(x_axis, y_price_med,label="€€", height = 0.3)
    ax2.barh(x_axis + 0.3,y_price_exp, label="€€€", height = 0.3)
    ax2.legend(fontsize = 20)
    ax2.yticks(x_axis,x_price, fontsize = 30)
    ax2.xticks(fontsize = 30)
    ax2.title(f'Price level of {choice_shop.capitalize()}s in each district', fontsize=35)

    return st.pyplot(fig)


def plot_hist(df, choice_shop, choice_district):
    fig = plt.figure(figsize=(15, 20))
    plt.hist(df['our_rating'], bins = 5)
    plt.title(f'Distribution of ratings of establishments selling {choice_shop.capitalize()}s in {choice_district}', fontsize=35)
    return st.pyplot(fig)
