#!/usr/bin/env python

from scipy.io import netcdf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D  
#from netCDF4 import Dataset
import numpy as np
import matplotlib.cm as cm

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

#dataset = Dataset('vortest_40_2.nc')
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


'''
fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis([0, np.sqrt(cells), 0, np.sqrt(cells)])
colors = cm.rainbow(np.linspace(0, 1, cells))
cmap = matplotlib.cm.get_cmap('hot')
randata=np.zeros(cells)
for i in range(cells):
	cellpos=np.zeros(4)
	cellpos[0]=pos[0][2*i]
	cellpos[1]=pos[tmax-1][2*i]
	cellpos[2]=pos[0][2*i+1]
	cellpos[3]=pos[tmax-1][2*i+1]

	for ii in range(2):
		if ((cellpos[2*ii]<4) and (cellpos[2*ii+1]>np.sqrt(cells)-4)):
			cellpos[2*ii+1]=cellpos[2*ii+1]-np.sqrt(cells)
		if ((cellpos[2*ii+1]<4) and (cellpos[2*ii]>np.sqrt(cells)-4)):
			cellpos[2*ii+1]=cellpos[2*ii+1]+np.sqrt(cells)

	randata[i]=(cellpos[1]-cellpos[0])**2+(cellpos[3]-cellpos[2])**2
	if(randata[i]>100):
		randata[i]=0
maxran=max(randata)


#for i in range(cells):
for i in [20,40,65,95,120,139,155,186,191,1,2]:
	xdata=[]
	ydata=[]
	for t in range(40,tmax):
		if(((abs(pos[t-1][2*i]-pos[t][2*i])<1) and (abs(pos[t-1][2*i+1]-pos[t][2*i+1])<1)) or t==0): 
			xdata.append(pos[t][2*i])
			ydata.append(pos[t][2*i+1])
		else:
			break

	line = Line2D(xdata,ydata,color=cmap(randata[i]/maxran))
	ax.add_line(line)
		
	#plt.plot([pos[1][2*i]],pos[1][2*i+1], marker='o', markersize=2, color="black")
plt.savefig('distance.png')
plt.clf()
print('end')
'''
'''
closeneighs=np.zeros(2*cells)
strat=np.zeros(cells)
msd=np.zeros(bins)
numbin=np.zeros(bins)

print(gridpoints)
print(len(conc[0]))

for i in range(cells):
	strat[i]=(np.floor(pos[0][2*i+1]/(np.sqrt(cells)/bins)))
	for j in range(cells):
		if(i!=j):
			r=(pos[0][2*i]-pos[0][2*j])**2+(pos[0][2*i+1]-pos[0][2*j+1])**2
			if((r<closeneighs[2*i+1]) or (closeneighs[2*i+1]==0)):
				closeneighs[2*i+1]=r
				closeneighs[2*1]=j
for z in range(bins):
	for x in range(cells):
		if strat[x]==z:
			locs=np.zeros(8)
			numbin[z]+=1
			locs[0]=pos[0][2*x]
			locs[1]=pos[tmax-1][2*x]
			locs[2]=pos[0][2*x+1]
			locs[3]=pos[tmax-1][2*x+1]
			locs[4]=pos[0][2*int(closeneighs[2*x])]
			locs[5]=pos[tmax-1][2*int(closeneighs[2*x])]
			locs[6]=pos[0][2*int(closeneighs[2*x])+1]
			locs[7]=pos[tmax-1][2*int(closeneighs[2*x])+1]

			for ii in range(4):
				if ((locs[2*ii]<4) and (locs[2*ii+1]>np.sqrt(cells)-4)):
					locs[2*ii+1]=locs[2*ii+1]-np.sqrt(cells)
				if ((locs[2*ii+1]<4) and (locs[2*ii]>np.sqrt(cells)-4)):
					locs[2*ii+1]=locs[2*ii+1]+np.sqrt(cells)




			msd[z]+=((locs[1]-locs[5])-(locs[0]-locs[4]))**2+((locs[3]-locs[7])-(locs[2]-locs[6]))**2

	if numbin[z]==0:
		numbin[z]=1
	msd[z]=msd[z]/numbin[z]

for i in range(bins):
	print(msd[i])

timeavec =np.zeros(int(gridpoints))
spaceavec=np.zeros(tmax)

for i in range(int(gridpoints)):
	for t in range(tmax):
		if t==0:
			pt=1
		else:
			pt=times[t-1]
		timeavec[i] += (times[t]-pt)*conc[t][i]/(times[tmax-1]-1)
		spaceavec[t] += conc[t][i]/(100*gridpoints)

print(spaceavec)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogx(times[range(tmax)],spaceavec[range(tmax)])
plt.xlabel('Time Step')
plt.ylabel('Average Concentration')
plt.savefig('spaceaverage.png')


timeavec.shape = (timeavec.size//int(np.ceil(np.sqrt(cells*10))) , int(np.ceil(np.sqrt(cells*10))))

plt.clf()
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(timeavec,cmap='inferno')
ax.set_aspect('equal')

cax = fig.add_axes([0, 0, np.sqrt(gridpoints), np.sqrt(gridpoints)])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.savefig('timeaverage.png')


'''

closeneighs=np.zeros(2*cells)
strat=np.zeros((len(pos),cells))
#strat=np.zeros(cells)


#1e6 every 1000 and 1e4 every 1
#tvals=[[10,10000],[20,5000],[50,2000],[100,1000],[200,500],[500,200],[1000,100],[2000,50],[5000,20],[10000,10]]
#1e6 every 10
#tvals=[[100,10000],[200,5000],[500,2000],[1000,1000],[2000,500],[5000,200],[10000,100],[20000,50],[50000,20],[100000,10]]
#1e6 every 100
tvals=[[100,10000],[200,5000],[500,2000],[1000,1000],[2000,500],[5000,200],[10000,100],[20000,50],[50000,20],[100000,10]]
##tvals=[[10,1000],[20,500],[50,200],[100,100],[200,50],[500,20],[1000,10],[2000,5],[5000,2],[10000,1]]



print(pos.max(1))
#P0 v S0 Plot
for t in range(0,len(pos),100):
	for i in range(cells):
		strat[t][i]=(np.floor(pos[t][2*i+1]/(np.sqrt(cells)/bins)))


meanp0array=np.zeros(bins)
means0array=np.zeros(bins)
meanspeedarray=np.zeros(bins)
meanvelarray=np.zeros((bins,2))
countarray=np.zeros(bins)


#convert=['black','purple','darkviolet','orange','gold','gold','orange','darkviolet','purple','black']
#fig = plt.figure()
#ax = fig.add_subplot(111)
for z in range(bins):
	#for t in range(0,len(pos),100):
	for t in range(0,len(pos),100):
		for x in range(cells):
			if strat[t][x]==z:
				meanp0array[z]+=3.5+.5*np.mean(cellconc[t][x])
				means0array[z]+=np.mean(s0[t][x])
				#meanspeedarray[z]+=np.sqrt((vel[t][2*x])**2+(vel[t][2*x+1])**2)
				#meanvelarray[z]+=[vel[t][2*x],vel[t][2*x+1]]
				countarray[z]+=1
				#plt.scatter(3.95-.2*np.mean(cellconc[t][x]),np.mean(s0[t][x]),c=convert[z])

#plt.scatter(3.8-.3*cellconc[0],s0[0])
#plt.xlabel('P0')
#plt.ylabel('S0')
#plt.savefig('pvp0.png')


for z in range(bins):
	print(str(meanp0array[z]/countarray[z])+' ' + str(means0array[z]/countarray[z]))
	#print(str(meanspeedarray[z]/countarray[z])+' ' + str(meanvelarray[z][0]/countarray[z])+ ' ' + str(meanvelarray[z][1]/countarray[z]))




'''

for i in range(cells):
	strat[i]=(np.floor(pos[0][2*i+1]/(np.sqrt(cells)/bins)))
	for j in range(cells):
		if(i!=j):
			r=(pos[0][2*i]-pos[0][2*j])**2+(pos[0][2*i+1]-pos[0][2*j+1])**2
			if((r<closeneighs[2*i+1]) or (closeneighs[2*i+1]==0)):
				closeneighs[2*i+1]=r
				closeneighs[2*i]=j


#MSD Data
for t in range(len(tvals)):
	numbin=np.zeros(bins)
	msd=np.zeros(bins)
	for z in range(bins):
		#for x in range(1):
		for x in range(cells):
			if strat[x]==z:		
				for tcur in range(tvals[t][1]):
					if tcur==0:
						tpast=0
					else:
						tpast=int(-1+tcur*(tvals[t][0]/100))
					tact=int(-1+(tcur+1)*(tvals[t][0]/100))
					#print(tpast)
					#print(tact)
					#print('end')

					locs=np.zeros(8)
					numbin[z]+=1
					locs[0]=pos[tpast][2*x]
					locs[1]=pos[tact][2*x]
					locs[2]=pos[tpast][2*x+1]
					locs[3]=pos[tact][2*x+1]
					locs[4]=pos[tpast][2*int(closeneighs[2*x])]
					locs[5]=pos[tact][2*int(closeneighs[2*x])]
					locs[6]=pos[tpast][2*int(closeneighs[2*x])+1]
					locs[7]=pos[tact][2*int(closeneighs[2*x])+1]

					for ii in range(4):
						if ((locs[2*ii]<4) and (locs[2*ii+1]>np.sqrt(cells)-4)):
							locs[2*ii+1]=locs[2*ii+1]-np.sqrt(cells)
						if ((locs[2*ii+1]<4) and (locs[2*ii]>np.sqrt(cells)-4)):
							locs[2*ii+1]=locs[2*ii+1]+np.sqrt(cells)
				
					msd[z]+=((locs[1]-locs[5])-(locs[0]-locs[4]))**2+((locs[3]-locs[7])-(locs[2]-locs[6]))**2

		if numbin[z]==0:
			numbin[z]=1
		msd[z]=msd[z]/numbin[z]

	for i in range(bins):
		print(msd[i])
'''

'''
#Solo MSD Data
for t in range(len(tvals)):
	numbin=np.zeros(bins)
	msd=np.zeros(bins)
	for z in range(bins):
		#for x in range(1):
		for x in range(cells):
			if strat[x]==z:		
				for tcur in range(tvals[t][1]):
					if tcur==0:
						tpast=0
					else:
						tpast=int(-1+tcur*(tvals[t][0]/100))
					tact=int(-1+(tcur+1)*(tvals[t][0]/100))
					#print(tpast)
					#print(tact)
					#print('end')

					locs=np.zeros(8)
					numbin[z]+=1
					locs[0]=pos[tpast][2*x]
					locs[1]=pos[tact][2*x]
					locs[2]=pos[tpast][2*x+1]
					locs[3]=pos[tact][2*x+1]
					locs[4]=0
					locs[5]=0
					locs[6]=0
					locs[7]=0

					for ii in range(4):
						if ((locs[2*ii]<4) and (locs[2*ii+1]>np.sqrt(cells)-4)):
							locs[2*ii+1]=locs[2*ii+1]-np.sqrt(cells)
						if ((locs[2*ii+1]<4) and (locs[2*ii]>np.sqrt(cells)-4)):
							locs[2*ii+1]=locs[2*ii+1]+np.sqrt(cells)
				
					msd[z]+=((locs[1]-locs[5])-(locs[0]-locs[4]))**2+((locs[3]-locs[7])-(locs[2]-locs[6]))**2

		if numbin[z]==0:
			numbin[z]=1
		msd[z]=msd[z]/numbin[z]

	for i in range(bins):
		print(msd[i])

'''