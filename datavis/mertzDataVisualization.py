#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd
import datetime as dt

#inputing the data
data = pd.read_csv('USvideos.csv', delimiter=',')

#defining the attributes for the plot
x = [dt.datetime.strptime(point, "%y.%d.%m") for point in data['trending_date']]
y = data['views']/1000
color = [a /(a+b+1) for a,b in zip(data['likes'],data['dislikes'])]
size = data['comment_count']/1000


'''
the scatter plot takes the following parameters:

x = Trending Date
y = Views (in thousands)
c = likes/dislikes+likes+1, (the +1 is to avoid devision by 0) 
    yeilding the ratio of likes to total ratings
    
cmap = a Red, Yellow, Green color map from matplotlib.cm

alpha = .5, makes the points transparent to better show parts 
        of the graph where there are many points
        

'''

fig = plt.scatter(x,y,s=size,c=color,cmap=cm.RdYlGn,alpha=.5)


cb = plt.colorbar()
cb.set_label("Likes / Total Ratings")

#Getting the first and last trending_date entries to scale the x axis
firstdate = dt.datetime.strptime(data['trending_date'][0], "%y.%d.%m")
lastdate = dt.datetime.strptime(data['trending_date'][20965], "%y.%d.%m")
plt.xlim(firstdate,lastdate)

#adjust for different sizes of figure
plt.rcParams["figure.figsize"] = (20,20)

#rotate the x-tick markers slightly to avoid overlap of dates
plt.xticks(rotation=15)

#axis labels and title
plt.xlabel("Trending Date")
plt.ylabel("Views \n(in Thousands)")
plt.suptitle('Trending Youtube Videos from November to February \n '+\
     '(Point Size = Comments in Thousands)')

plt.show()