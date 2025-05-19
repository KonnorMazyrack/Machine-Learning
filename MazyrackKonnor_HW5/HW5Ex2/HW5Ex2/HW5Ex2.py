from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly
import plotly.graph_objs as go

data = pd.read_csv('vehicles.csv')
data = data.drop(['make'], axis=1)

markersize = data['qsec']
markercolor = data['mpg']
markershape = data['am'].map({1:'square', 0: 'circle'})

fig1 = go.Scatter3d(x=data['wt'],
                    y=data['disp'],
                    z=data['hp'],
                    marker=dict(size=markersize,
                                color=markercolor,
                                symbol=markershape,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

mylayout = go.Layout(scene=dict(xaxis=dict( title="wt"),
                                yaxis=dict( title="disp"),
                                zaxis=dict( title="hp")),)

plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("6DPlot.html"))

