'''
Find local value of g_c for each timestep/pltfile from

1. create a slice of x velocity at dx/2 up from the wall
2. clip slice to relevant area for averaging
3. save all timestep data
4.  
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py




def DERIV(x,y):
	# quick little first derivative function
	# use centered differnce without constant spacing
	dydx = np.zeros([len(x),1])
	for i in range(0,len(x)-1):
		dydx[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])

	dydx[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])

	return dydx

def SMA(data, window):
	# smoothed moving averaged
	# smooth data within window
    triangle=np.concatenate((np.arange(window + 1), np.arange(window)[::-1]))
    smoothed=[]

    for i in range(window, len(data) - window * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(window + window/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed



# change the directory
path ="/media/u0890475/6f7f5b18-6951-4d23-9e1b-146d3d4c2671/Matt_Old/phi375_amr3_plt/plt/"
#path = "/media/u1474458/Data/phi375_amr3_plt/plt/"
U_title = "U_wall_phi375" # name of all CSV files before timestep
Flame_title = "CH2_0.8_phi375" # name of all CSV files before timestep


Xf = [] # x coordinate of leading point on flame surface
Yf = [] # y ''
Zf = [] # z ''
L_sz = [] # length of sepearation zone

num_timesteps = 175

for i in range(0,num_timesteps):
	if i == 84 or i == 84: # number 84 broke somehow ... not sure how
		continue
	print(i)


	suffix = "_" + str(i)


	Uinfile = path + U_title + suffix + ".csv"
	Finfile = path + Flame_title + suffix + ".csv"
	f = open(Finfile, "r")
	f2 = open(Uinfile,"r")


	##---------------- Everything for the leading point of the flame ---------------------------

	# first thing to do, read the first line and identify number of columns
	header = f.readline()
	header2 = header.split(",") # make a list of the header
	


	# find column corresponding to X points
	X_col = header2.index('"Points:0"')
	Y_col = header2.index('"Points:1"')
	U_col = header2.index('"x_velocity"\n')

	# load columns into variables
	X_vals = np.loadtxt(Finfile,usecols=[X_col],delimiter=",",skiprows = 1) # all the x values of the surface/inputfile
	Y_vals = np.loadtxt(Finfile,usecols=[Y_col],delimiter=",",skiprows = 1) # all the x values of the surface/inputfile
	U = np.loadtxt(Finfile,usecols=[U_col],delimiter=",",skiprows = 1) # all the x_velocities ^^	^^	^^	^^	^^

	# find index of minimum (closest to inlet) of x vals
	ind = X_vals.argmin()

	# X,Y,Z values of the flame forward point
	Xf.append(float(X_vals[ind]))
	Yf.append(float(Y_vals[ind]))
	X_vel.append(float(U[ind]))
	
	##------------------ X-Velocity ahead of the flame -----------------------------------

	# first thing to do, read the first line and identify number of columns
	header = f2.readline()
	header2 = header.split(",") # make a list of the header

	# find column corresponding to X points
	X_col = header2.index('"Points:0"')
	Y_col = header2.index('"Points:1"')
	Z_col = header2.index('"Points:2"')
	U_col = header2.index('"x_velocity"\n')

	# load columns into variables
	X_vals = np.loadtxt(Uinfile,usecols=[X_col],delimiter=",",skiprows = 1) # all the x values of the surface/inputfile
	Y_vals = np.loadtxt(Uinfile,usecols=[Y_col],delimiter=",",skiprows = 1) # all the y values of the surface/inputfile
	Z_vals = np.loadtxt(Uinfile,usecols=[Z_col],delimiter=",",skiprows = 1) # all the z values of the surface/inputfile
	U = np.loadtxt(Uinfile,usecols=[U_col],delimiter=",",skiprows = 1) # all the x_velocities ^^	^^	^^	^^	^^
	'''
	# METHOD 1: 5 quenching distance ahead of flame
	# use this because its relatively easy to compute
	# calculate point N quencing distances ahead of the flame
	delta = 5*Yf[-1]
	
	X_mes = Xf[-1] - delta

	diff = abs(X_vals-X_mes)
	ind = diff.argmin()
	g.append(U[ind]/Y_vals[ind])

	'''
	# METHOD 2: N separation zones ahead
	# use this method because it is more rigorous
	# calculate forward location of separation zone

	R = np.where(U<=0,1,0)
	X_sz = R*X_vals
	Y_sz = R*Y_vals
	Z_sz = R*Z_vals
	xback = min(X_vals[np.where(X_sz>0)])
	L_sz.append(max(X_sz)-xback) # length of the separation zone
	#print(L_sz[-1])

	delta = 1.25*L_sz[-1]
	X_mes = Xf[-1] - delta


	diff = abs(X_vals-X_mes)
	ind = diff.argmin()
	g.append(U[ind]/Y_vals[ind])
	
	

	# where U is less that zero (inside separation bubble)
	# return index of largest ans smallest X value


	## test if this method is working by plotting points with U<0
	#print(xback)
	#plt.plot(X_sz,Z_sz,'o')
	#plt.show()



	##--------------------------------------------------------------------------------------

	row1 = f.readline().split(",") # make a list of the first row so that I can get the time coordinate
	t.append(float(row1[0])) # append the time of the timestep for everything


	f.close()
	f2.close()

# get expected value of g based on Novo model
C = h5py.File(gc_path)
gC = np.array(C.get('g_c'))
phiC = np.array(C.get('phi'))
aC= np.array(C.get('alpha'))
SfC = np.array(C.get('S'))





DIFF = abs(.375-phiC)
ind = np.where(DIFF == min(DIFF))
g_FB = gC[ind]
a_FB = aC[ind]
S_FB = SfC[ind]


S = DERIV(t,Xf)
S_f = X_vel[:,1]-S[:] # flame speed

print(np.shape(S),np.shape(X_vel))

g_diff = g-g_FB
SIGN = S*g_diff


#g = SMA(g,3) # smooth the dataset g 


dt = np.sqrt(a_FB/g)
dc = S_FB/g



M = S_FB**2/(a_FB*g)



ratio = np.mean(DERIV(dt,dc))
print(ratio)


for i in range(0,len(dt)):
	plt.plot(dt[i],dc[i],'or')
	txt = str(t[i])
	plt.annotate(txt,[dt[i],dc[i]])

plt.plot(dt,dc,'or')
plt.title("d_t vs d_c")
plt.xlabel("d_t")
plt.ylabel("d_c")
plt.plot(dt,dt,'-k')
plt.figure()







#plt.plot(phiC,gC)
#plt.figure()
plt.plot(t,g,'-or')
plt.plot(t,np.ones([len(t),1])*g_FB,'-r')
plt.ylabel("g    [s^-1]")
plt.legend(["g","g_FB"])
plt.twinx()
plt.plot(t,Xf,'--ob')
plt.ylabel("Flame Position")
plt.xlabel("time")
plt.figure()



plt.plot(t,Yf,'-or')
#plt.plot(t,dc*np.ones([len(t),1]),'-k')
plt.plot(t,dt,'-k')
plt.plot(t,dc,'-b')
plt.legend(["Flame tip Y","d thermal", "d critical"])
plt.xlabel("time")
plt.ylabel("Y distance")




plt.figure()
plt.plot(t,Xf,'-or')
plt.ylabel("Flame Tip X")
plt.xlabel("time")
plt.twinx()
plt.plot(t,dt,'-k')
plt.plot(t,dc,'-b')
plt.legend(["d thermal", "d critical"])
plt.ylabel("Y distance")


plt.figure()
plt.plot(t,Xf,'-or')
plt.ylabel("Flame Tip X")
plt.xlabel("time")
plt.twinx()
plt.plot(t,M,'-b')
plt.plot(t,np.ones([len(t),1]),'--k')
plt.ylabel("S^2/ag")









plt.show()