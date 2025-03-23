'''
this code used to locate X location in a CSV file output from paraview
also create recession/advance frequency plot(S vs U in the lab frame)

to use this

first save all timesteps in paraview with data desired
this generally looks like
cell -> point data
contour H2 progress variable = 0.5 or whatever we want
clip to relevant region
save data as csv with all timesteps using a _%d extension for timesteps


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def DERIV(x,y):
	# quick little first derivative function
	# use centered differnce without constant spacing
	dydx = np.zeros([len(x),1])
	for i in range(0,len(x)-1):
		dydx[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])

	dydx[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])

	return dydx




#path = "/home/u1474458/CF_stuff/AMR2_to_FB/plt_results/phi375_amr3/plt/" # path to folder with all timestep CSV files
path = "/media/u1474458/Data/phi375_amr3_plt/plt/"
title = "CH2_0.5_phi375" # name of all CSV files before timestep


t = []
X_max = []
X_vel = []


num_timesteps = 176

for i in range(0,num_timesteps):
	if i == 84: # number 84 broke somehow ... not sure how
		continue

	suffix = "_" + str(i)
	infile = path + title + suffix + ".csv"
	f = open(infile, "r")
	# first thing to do, read the first line and identify number of columns
	header = f.readline()
	header2 = header.split(",") # make a list of the header
	print(i)
	# find column corresponding to X points
	X_col = header2.index('"Points:0"')
	U_col = header2.index('"x_velocity"\n')

	# load columns into variables
	X_vals = np.loadtxt(infile,usecols=[X_col],delimiter=",",skiprows = 1) # all the x values of the surface/inputfile
	U = np.loadtxt(infile,usecols=[U_col],delimiter=",",skiprows = 1) # all the x_velocities ^^	^^	^^	^^	^^

	# find index of minimum (closest to inlet) of x vals
	ind = np.argmin(X_vals)

	X_max.append(float(.12-X_vals[ind])) 
	X_vel.append(float(U[ind]))
	
	row1 = f.readline().split(",") # make a list of the first row so that I can get the time coordinate
	t.append(float(row1[0])) # append the time of the timestep for everything


	f.close()


S = DERIV(t,X_max) # flame speed in the lab frame (advance vs recede)
Sf = S+X_vel # the true flame speed



plt.plot(t,X_max)

plt.xlabel("time [s]")
plt.ylabel("Forward Flame Location from Isothermal wall")
plt.xticks(np.linspace(0,max(t),10))
plt.tick_params(direction="in", left="off",labelleft="on")


plt.twinx()
plt.plot(t,S,'r')
plt.plot(t,np.zeros([len(t),1]),'--r') # add x axis
plt.ylabel("Leading Point Advance Speed")



plt.figure()
plt.plot(t,S,'r')
#plt.ylabel("Leading Point Advance Speed")
#plt.twinx()
plt.plot(t,X_vel,'b')
plt.plot(t,np.zeros([len(t),1]),'--') # add x axis
#plt.ylabel("X velocity at Leading Point")
plt.xlabel("time")
plt.legend(["Lab S_f","X velocity"])
plt.tick_params(direction="in", left="off",labelleft="on")
#plt.show()

'''
key for signs of data

+ S is advancing flame
+ X_vel is flow towards outlet

so the two + are in different directions

Q1 - FB against U mean (highly strained flame?)
Q2 - FB with back-flow region (FB expected)
Q3 - recession with backflow region (highly strained flame?)
Q4 - recession with U mean (blow off)


'''

## create the frequency plot
# x axis is x-vel at the leading point
# y axis is S in lab frame (recede/advance)
plt.figure()
for i in range(0,len(S)):
	plt.plot(X_vel[i],S[i],'or')

## add x,y axes
plt.plot(X_vel,np.zeros([len(t),1]),'-k')
plt.plot(np.zeros([len(t),1]),S,'-k')
plt.xlabel("X Velocity")
plt.ylabel("Lab Flame Speed")
plt.title("Frequency Plot for S_lab and U")
plt.tick_params(direction="in", left="off",labelleft="on")


plt.show()

'''
# animate plots
x = t
y = X_max
N = len(t)

fig,ax = plt.subplots()
plt.plot(x,y)

line = ax.plot(x[0],y[0],'o')[0]
def update(frame):
	x0 = x[frame]
	y0 = y[frame]
	line.set_xdata(x0)
	line.set_ydata(y0)
	return line

ani = animation.FuncAnimation(fig = fig,func = update,frames = N, interval = 1000)
ani.save(filename="tvsX.gif", writer="ffmpeg")
plt.show()
'''