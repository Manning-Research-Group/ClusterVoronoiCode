#!/usr/bin/env python

from scipy.io import netcdf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D  
import numpy as np
import matplotlib.cm as cm

dataset = netcdf.netcdf_file('vortest_800_35t4_4.nc', mode='r')
pos = dataset.variables['pos'][:]
conc = dataset.variables['Concentration'][:]
cellconc= dataset.variables['Cell Concentration'][:]
vel = dataset.variables['Calculated Cell Velocity'][:]
s0 = dataset.variables['s0'][:]
times = dataset.variables['time'][:]
cells=800
bins=10
gridpoints =  (np.ceil(np.sqrt(cells*10)))**2
tmax=len(pos)

closeneighs=np.zeros(2*cells)
strat=np.zeros(cells)



numsteps=2730000
intsteps=10000
dt=0.0001


gapstep=int(10/(intsteps*dt))
print((numsteps-gapstep))
avesteps=numsteps/gapstep
velmat=np.zeros((int(np.ceil((cells*(numsteps/intsteps)))),3))
#print(int(gapstep-1))
#Create Instanteous Velocity matrix
for x in range(cells):
	for t in range(int(gapstep-1),int(numsteps*dt)):
		#print(t)
		velmat[x*t][0]=cellconc[t][x]
		velmat[x*t][2]=(np.floor(pos[t][2*x+1]/(np.sqrt(cells)/bins)))
		#print(pos[t][2*x+1])
		tact=t
		tpast=t-gapstep
		locs=np.zeros(8)
		locs[0]=pos[tpast][2*x]
		locs[1]=pos[tact][2*x]
		locs[2]=pos[tpast][2*x+1]
		locs[3]=pos[tact][2*x+1]


		for ii in range(4):
			if ((locs[2*ii]<4) and (locs[2*ii+1]>np.sqrt(cells)-4)):
				locs[2*ii+1]=locs[2*ii+1]-np.sqrt(cells)
			if ((locs[2*ii+1]<4) and (locs[2*ii]>np.sqrt(cells)-4)):
				locs[2*ii+1]=locs[2*ii+1]+np.sqrt(cells)
				
		velmat[x*t][1]=((locs[1]-locs[5])-(locs[0]-locs[4]))**2+((locs[3]-locs[7])-(locs[2]-locs[6]))**2

numbin=np.zeros(bins)		
msd=np.zeros(bins)
for i in range(bins):
	for x in range(cells):
		for t in range(int(gapstep-1),int(numsteps*dt)):
			if velmat[x*t][2]==i:		
				msd[i]+=velmat[x*t][1]
				numbin[i]+=1
			
print(numbin)
for i in range(bins):
	if numbin[i]==0:
		numbin[i]=1
	msd[i]=msd[i]/numbin[i]
	print(msd[i])

