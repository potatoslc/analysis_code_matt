'''
Read in data from the .dat files produced in PA_contours.py
Read in data from the .csv files produced by Paraview

This is the analysis section code:
 - determine leading point of flame in x,y,z posittion from isocontour (PeleAnalysis)
 - acquire near-wall velocity data from the slice in Paraview


'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

#------------------- File Management --------------------
# where are these files
gc_path = "/home/u1474458/Documents/BLF_Model_Matts/h5_results/T_300_P_100v1.h5" # path to g_c vs phi data (precalculated with model Sl^2/ag = 1)

outdir = "/media/u1474458/Data/phi375_amr3_plt/Analysis/" # where to put results (stored as .h5 dicts)
DAT_folder = "/media/u1474458/Data/phi375_amr3_plt/Analysis/DAT_files/"	# where the .dat files are stored
CSV_folder = "/media/u1474458/Data/phi375_amr3_plt/plt/" # where the .csv files are stored
CURV_folder = "/media/u1474458/Data/phi375_amr3_plt/Analysis/Curvature/DAT_files/" # where the curvature .dat isocontours are stored



SD = "phi375" # what to call Saved Data in h5 data structure
dat_prefix = "CH2_0.8"
U_prefix = "U_wall_phi375"
K_prefix = "CH2_0.8"

phi = 0.375
#----------------- Sort and List Directories---------------
# might not get all the files 
U = os.listdir(CSV_folder)
for i in range(len(U)-1,-1,-1):
	if U_prefix not in U[i]:
		del U[i]

F = os.listdir(DAT_folder)
for i in range(len(F)-1,-1,-1):
	if dat_prefix not in F[i]:
		del F[i]


K = os.listdir(CURV_folder)
for i in range(len(K)-1,-1,-1):
	if K_prefix not in K[i]:
		del K[i]
#-----------------Flame Contour Analysis (PeleAnalysis) -------------------
t = []
Xf = []
Yf = []
Zf = []
Uf = []
g = []
L_sz = []
kf = []
Gkf = []
for i in range(len(F)):
	if i == 85: # issue with phi375 files for wahtever reason
		continue
	print(i)

	file = DAT_folder + dat_prefix + "_" + str(i) + ".dat"

	f = open(file,"r")
	header = f.readline().split(" ")
	X_col = header.index('X')-2 # subtract 2 for the header elements "VARIABLES" "="
	Y_col = header.index('Y')-2 # subtract 2 for the header elements "VARIABLES" "="
	Z_col = header.index('Z')-2 # subtract 2 for the header elements "VARIABLES" "="
	U_col = header.index('x_velocity')-2 # subtract 2 for the header elements "VARIABLES" "="
	
	# get some important information in line 2
	line2 = f.readline().split(" ")
	t.append(float(line2[1].split('"')[1])) # get the time stamp as a float number
	N = int(line2[2].split("=")[1]) # get the number of elements/rows we have to care about
	
	
	X_vals = np.loadtxt(file,usecols=[X_col],delimiter=" ",skiprows = 2,max_rows = N) # all the x values of the surface/inputfile
	Y_vals = np.loadtxt(file,usecols=[Y_col],delimiter=" ",skiprows = 2,max_rows = N) # all the y values of the surface/inputfile
	Z_vals = np.loadtxt(file,usecols=[Z_col],delimiter=" ",skiprows = 2,max_rows = N) # all the y values of the surface/inputfile
	U_vals = np.loadtxt(file,usecols=[U_col],delimiter=" ",skiprows = 2,max_rows = N) # all the x_velocities ^^	^^	^^	^^	^^

	ind = X_vals.argmin() # index of the row of the forward most point in the X direction
	#inspect the isosurface
	#fig = plt.figure()
	#ax = fig.add_subplot(projection='3d')
	#ax.plot_trisurf(X_vals,Z_vals,Y_vals)
	#plt.xlabel('X')
	#plt.ylabel('Z')
	#plt.show()

	# X,Y,Z values of the flame forward point
	Xf.append(float(X_vals[ind]))
	Yf.append(float(Y_vals[ind]))
	Zf.append(float(Z_vals[ind]))
	Uf.append(float(U_vals[ind]))

	f.close()

#------------------ Velocity Slice Analysis (Paraview)----------------------
	file = CSV_folder + U_prefix + "_" + str(i) + ".csv"
	f = open(file,"r")

	header2 = f.readline().split(",") # make a list of the header

	# find column corresponding to X points
	X_col = header2.index('"Points:0"')
	Y_col = header2.index('"Points:1"')
	Z_col = header2.index('"Points:2"')
	U_col = header2.index('"x_velocity"\n')

	# load columns into variables
	X_vals = np.loadtxt(file,usecols=[X_col],delimiter=",",skiprows = 1) # all the x values of the surface/inputfile
	Y_vals = np.loadtxt(file,usecols=[Y_col],delimiter=",",skiprows = 1) # all the y values of the surface/inputfile
	Z_vals = np.loadtxt(file,usecols=[Z_col],delimiter=",",skiprows = 1) # all the z values of the surface/inputfile
	U_vals = np.loadtxt(file,usecols=[U_col],delimiter=",",skiprows = 1) # all the x_velocities ^^	^^	^^	^^	^^
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

	R = np.where(U_vals<=0,1,0)
	X_sz = R*X_vals
	Y_sz = R*Y_vals
	Z_sz = R*Z_vals
	xback = min(X_vals[np.where(X_sz>0.01)]) # forward most point x_vel < 0

	L_sz.append(Xf[-1]-xback) # length of the separation zone

	delta = 5*L_sz[-1]
	X_mes = Xf[-1] - delta

	
	## test if this method is working by plotting points with U<0
	'''
	print(L_sz[-1])
	plt.plot(X_sz,Z_sz,'o')
	plt.plot(Xf[-1],Zf[-1],'or')
	plt.plot(xback,Zf[-1],'ok')
	plt.show()
	'''
	diff = abs(X_vals-X_mes)
	ind = diff.argmin()
	g.append(U_vals[ind]/Y_vals[ind])

	# check that times match
	line2 = f.readline().split(",")
	tcheck = float(line2[0])
	'''
	if not abs(t[-1] - tcheck) < 5e-8:
		print(t[-1])
		print(tcheck)
		print("times do not agree at timestep " + str(i))
		#sys.exit(1)

	'''

	f.close()



#---------------- Curvature --------------------------------------
	file = CURV_folder + K_prefix + "_" + str(i) + ".dat"
	f = open(file,"r")
	header = f.readline().split(" ")

	X_col = header.index('X')-2 # subtract 2 for the header elements "VARIABLES" "="
	Y_col = header.index('Y')-2 # subtract 2 for the header elements "VARIABLES" "="
	Z_col = header.index('Z')-2 # subtract 2 for the header elements "VARIABLES" "="
	k_col = header.index('MeanCurvature_Y(H2)')-2 # subtract 2 for the header elements "VARIABLES" "="
	Gk_col = header.index('GaussianCurvature_Y(H2)\n')-2 # subtract 2 for the header elements "VARIABLES" "="
	
	# get some important information in line 2
	line2 = f.readline().split(" ")
	N = int(line2[2].split("=")[1]) # get the number of elements/rows we have to care about
	
	
	X_vals = np.loadtxt(file,usecols=[X_col],delimiter=" ",skiprows = 2,max_rows = N) # all the x values of the surface/inputfile
	Y_vals = np.loadtxt(file,usecols=[Y_col],delimiter=" ",skiprows = 2,max_rows = N) # all the y values of the surface/inputfile
	Z_vals = np.loadtxt(file,usecols=[Z_col],delimiter=" ",skiprows = 2,max_rows = N) # all the y values of the surface/inputfile
	k_vals = np.loadtxt(file,usecols=[k_col],delimiter=" ",skiprows = 2,max_rows = N) # all the x_velocities ^^	^^	^^	^^	^^
	Gk_vals = np.loadtxt(file,usecols=[Gk_col],delimiter=" ",skiprows = 2,max_rows = N) # all the x_velocities ^^	^^	^^	^^	^^


	ind = X_vals.argmin()

	kf.append(k_vals[ind])
	Gkf.append(Gk_vals[ind])


	f.close()



	
#----------------- Related Quantities ---------------------------


# retrieve data from already calculated quantities
C = h5py.File(gc_path)
gC = np.array(C.get('g_c'))
phiC = np.array(C.get('phi'))
aC= np.array(C.get('alpha'))
SfC = np.array(C.get('S'))


DIFF = abs(phi-phiC) # phi of interest at the moment
ind = np.where(DIFF == min(DIFF))
g_FB = gC[ind]
a_FB = aC[ind]
S_FB = SfC[ind]

print("U at FB = " + str(g_FB*39.0625e-6))

# calcualted related quantities
dt = np.sqrt(a_FB/g)
dc = S_FB/g

M = S_FB**2/(a_FB*g)










#-------------------- Save Data in h5 data structure --------------


hf = h5py.File(outdir + SD + ".h5","w")
hf.create_dataset('Xf', data = Xf)
hf.create_dataset('Yf', data = Yf)
hf.create_dataset('Zf', data = Zf)
hf.create_dataset('Uf', data = Uf)
hf.create_dataset('g', data = g)
hf.create_dataset('dc',data = dc)
hf.create_dataset('dt',data = dt)
hf.create_dataset('t',data = t)
hf.create_dataset('kf', data = kf)
hf.create_dataset('Gkf', data = Gkf)
hf.close()



print("Data was Saved to " + str(outdir+SD+".h5"))

#-----------------------Plotting-----------------------------------
plt.plot(t,Xf,'-or')
plt.xlabel("time")
plt.ylabel("X coordinate")
plt.title("X vs t")




plt.figure()
plt.plot(t,Yf,'-or')
plt.xlabel("time")
plt.ylabel("Y coordinate")
plt.title("Y vs t")


plt.figure()
plt.plot(t,Xf,'-or')
plt.ylabel("Flame Tip X")
plt.xlabel("time")
plt.twinx()
plt.plot(t,dt,'-k')
plt.plot(t,dc,'-b')
plt.legend(["d thermal", "d critical"])
plt.ylabel("Y distance")

plt.show()
