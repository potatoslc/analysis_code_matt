'''
plot the results of data extraction in PA_PV.py

'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors.to_rgba as to_rgba
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
    smoothed = np.array(smoothed)
    return smoothed








f = h5py.File("/media/u1474458/Data/phi375_amr3_plt/Analysis/phi375.h5","r")
f2 = h5py.File("/media/u1474458/Data/phi40_amr3_plt/Analysis/phi40.h5","r")
print(f.keys())

Xf1 = np.array(f['Xf'])
Yf1 = np.array(f['Yf'])
Uf1 = np.array(f['Uf'])
dt1 = np.array(f['dt'])
dc1 = np.array(f['dc'])
g1 = np.array(f['g'])
t1 = np.array(f['t'])
kf1 = np.array(f['kf'])
Gkf1 = np.array(f['Gkf'])


Xf2 = np.array(f2['Xf'])
Yf2 = np.array(f2['Yf'])
Uf2 = np.array(f2['Uf'])
dt2 = np.array(f2['dt'])
dc2 = np.array(f2['dc'])
g2 = np.array(f2['g'])
t2 = np.array(f2['t'])
kf2 = np.array(f2['kf'])
Gkf2 = np.array(f2['Gkf'])

t = np.concatenate((t1,t2))
g = np.concatenate((g1,g2))
Xf = np.concatenate((Xf1,Xf2))
Yf = np.concatenate((Yf1,Yf2))
Uf = np.concatenate((Uf1,Uf2))
dc = np.concatenate((dc1,dc2))
dt = np.concatenate((dt1,dt2))
kf = abs(np.concatenate((kf1,kf2)))
Gkf = np.concatenate((Gkf1,Gkf2))

t = t[50:220]
g = g[50:220]
Xf = Xf[50:220]
Yf = Yf[50:220]
Uf = Uf[50:220]
dc = dc[50:220]
dt = dt[50:220]
kf = kf[50:220]
Gkf = Gkf[50:220]







M = dc**2/dt**2 # sl^2/ag

Sf = -DERIV(t,Xf)[:,0] + Uf # flame speed
AR1 = -DERIV(t1,Xf1)[:,0] # advance rate
AR2 = -DERIV(t2,Xf2)[:,0] # advance rate
AR = -DERIV(t,Xf)[:,0] # advance rate










#--------- get flashback data -----------
gc_path = "/home/u1474458/Documents/BLF_Model_Matts/h5_results/T_300_P_100v1.h5" # path to g_c vs phi data (precalculated with model Sl^2/ag = 1)

C = h5py.File(gc_path)
gC = np.array(C.get('g_c'))
phiC = np.array(C.get('phi'))
aC= np.array(C.get('alpha'))
SfC = np.array(C.get('S'))

#plt.plot(phiC,gC,'o')
#plt.figure()
DIFF = abs(.40-phiC) # .40 because that is the phi of interest at the moment
ind = np.where(DIFF == min(DIFF))
g40 = gC[ind]
a40 = aC[ind]
S40 = SfC[ind]

DIFF = abs(.37-phiC) # .375 because that is the phi of interest at the moment
ind = np.where(DIFF == min(DIFF))
g375 = gC[ind]
a375 = aC[ind]
S375 = SfC[ind]

#----------- dc and dt ----------------


t_crit = t1[-1] + .12/8 # when the inlet flux change reaches the flame front (roughly)
t11 = []
t22 = []
for i in range(len(t)):
	if t[i] < t_crit:
		t11.append(t[i])
		ind = i+1
	else:
		t22.append(t[i])



dc375 = S375/g[:ind]
dt375 = np.sqrt(a375/g[:ind])
dc40 = S40/g[ind:]
dt40 = np.sqrt(a40/g[ind:])

dc2 = np.concatenate((dc375,dc40))
dt2 = np.concatenate((dt375,dt40))

M2 = dc2/dt2
M2 = SMA(M2,2)




AR1 = -DERIV(t11,Xf[:ind])[:,0] # advance rate
AR2 = -DERIV(t22,Xf[ind:])[:,0] # advance rate
#------------ Plot -------------------- 
'''
plt.plot(t,Xf,'-or')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Xf),min(Xf2),len(t)),'-k')
plt.ylabel("X coordinate")
plt.xlabel("time")
plt.twinx()
g = SMA(g,3)
plt.plot(t,g,'-b')

plt.ylabel("g")
plt.ylim([0,50e3])

plt.title("X vs t")



plt.figure()
plt.plot(t,Yf,'-or')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Yf),min(Yf),len(t)),'-k')
plt.xlabel("time")
plt.ylabel("Y coordinate")
plt.title("Y vs t")


plt.figure()
plt.plot(t,Xf,'-or')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Xf),min(Xf2),len(t)),'-k')
plt.ylabel("Flame Tip X")
plt.xlabel("time")
plt.twinx()
#dt = SMA(dt,3)
#dc = SMA(dc,3)
plt.plot(t,dt,'-k')
plt.plot(t,dc,'-b')

plt.legend(["d thermal", "d critical"])
plt.ylabel("Y distance")


plt.figure()
#Uf = SMA(Uf,7)
plt.plot(t,Uf,'-ob')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Uf),min(Uf),len(t)),'--b')
plt.plot(t,np.zeros([len(t),1]),'-k')
plt.twinx()
plt.plot(t,Xf,'-or')

plt.title("U and X at Flame Tip vs time")



plt.figure()
M = SMA(M,7)
plt.plot(t,M, '-ob')
plt.ylabel("S^2/ag")
plt.plot(t,np.ones([len(t),1]),'-b')
plt.twinx()
plt.plot(t,Xf,'-or')
plt.ylabel("Flame X position")

plt.xlabel("time [s]")



plt.figure()
plt.title("g_mes and g flashback")
plt.plot(t,g,'r')
plt.plot(t1,g375*np.ones([len(t1),1]))
plt.plot(t2,g40*np.ones([len(t2),1]))


plt.figure()
plt.title("Flame Speed")
plt.plot(t,Sf)
plt.plot(t,AR)
plt.legend(["Flame Speed", "Advance Rate"])
plt.plot(t,np.zeros([len(t),1]),'k')


ax = plt.figure()
kf = SMA(kf,3)
plt.plot(t,kf)
plt.ylabel("Curvature")
plt.twinx()
plt.plot(t,Xf,'-r')
plt.ylabel('Flame Position')
ax.legend(["Flame position", "Curvature"])

plt.figure()
plt.plot(t,Xf,'-r')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Xf),min(Xf2),len(t)),'-k')
plt.plot(np.ones([len(t),1])*t_crit, np.linspace(max(Xf),min(Xf2),len(t)),'--k')
plt.twinx()
plt.plot(t11,dt375,'-k')
plt.plot(t11,dc375,'-b')
plt.plot(t22,dt40,'-k')
plt.plot(t22,dc40,'-b')
plt.text(.06,.80*max(dc375),"Phi = 0.375",fontsize = 18)
plt.text(.135,.80*max(dc375),"Phi = 0.40",fontsize = 18)
#plt.text(.135,.75*max(dc375),"Flashback",fontsize = 18)

'''
'''
ax = plt.figure(figsize = (9,6))
plt.plot(t[:220],Xf[:220],'-r')
plt.ylabel("Flame X Position [m]",fontsize = 18)
plt.xlabel("Time [s]", fontsize = 18)
plt.tick_params(direction="in", left="off",labelleft="off")
plt.plot(np.ones([len(t[:220]),1])*t_crit, np.linspace(max(Xf),min(Xf2),len(t[:220])),'-k')
plt.text(.09,.118,"ER = 0.375",fontsize = 18)
plt.text(.135,.118,"ER = 0.40",fontsize = 18)
plt.twinx()
plt.plot(t[:220],M2[:220],'-b')
plt.tick_params(direction="in")
plt.ylabel(r'$\frac{d_c}{d_t}$',fontsize = 18)
plt.plot(t[:220],np.ones([len(t[:220]),1]),'--b')
ax.legend(["Flame X Position", "_",  r'$\frac{d_c}{d_t}$'],fontsize = 12, loc = 1)


ax = plt.figure(figsize = (9,6))
tcut = t[:220]
plt.plot(t[:220],Xf[:220],'-r')
plt.plot(np.ones([len(t[:220]),1])*t_crit, np.linspace(max(Xf),min(Xf2),len(t[:220])),'-k')
plt.xlabel("Time [s]", fontsize = 18)
plt.text(.09,.118,"ER = 0.375",fontsize = 18)
plt.text(.135,.118,"ER = 0.40",fontsize = 18)
plt.tick_params(direction="in")
plt.ylabel("Flame X Position [m]",fontsize = 18)
plt.twinx()
plt.plot(t[:220],SMA(g[:220],5),'-k')
plt.tick_params(direction="in")
plt.plot(tcut,g375*np.ones([len(tcut),1]),'--k')
plt.ylabel("$g_c$",fontsize = 18)
plt.xlabel("Time [s]", fontsize = 18)
plt.text(.08,26200,"$g_{FB}$", fontsize = 18)
ax.legend(["Flame X Position", "_", "$g_c$"],fontsize = 12, loc = 1)




ax = plt.figure(figsize = (9,6))
plt.plot(tcut,SMA(Sf[:220],5),'-r')
plt.plot(tcut,SMA(Uf[:220],5),'-k')
plt.tick_params(direction="in")
plt.ylabel("Speed [m/s]", fontsize = 18)
plt.xlabel("Time [s]", fontsize = 18)
ax.legend(["Flame Speed", "X Velocity"],fontsize = 12, loc = 1)
plt.plot(np.ones([len(t[:220]),1])*t_crit, np.linspace(max(SMA(Sf,3)),min(Sf),len(t[:220])),'-k')
plt.plot(tcut,np.zeros([len(tcut),1]),'-k')
plt.grid(visible = 1)
plt.text(.09,1.1175,"ER = 0.375",fontsize = 18)
plt.text(.135,1.1175,"ER = 0.40",fontsize = 18)



ax = plt.figure(figsize = (9,6))
plt.plot(AR1,Uf[:ind],'or')
plt.plot(AR2,Uf[ind:],'ob')
ax.legend(["Stable Flame", "Flashback Flame"], fontsize = 12)
plt.plot(np.zeros([10,1]),np.linspace(-10,10,10),'k')

plt.plot(np.linspace(-10,10,10),np.zeros([10,1]),'k')
plt.ylim([1.1*min(Uf),1.1*max(Uf)])
plt.xlim([1.1*min(AR),1.1*max(AR)])
plt.text(0.25,2.0, "U > 0, AR > 0",fontsize = 18)
plt.text(-0.75,2.0, "U > 0, AR < 0",fontsize = 18)
plt.tick_params(direction="in")



## create that stupid density plot
X = np.linspace(-0.5,0.5,50)
Y = np.linspace(-0.5,0.5,50)

dx = X[2]-X[1]
dy = Y[2]-Y[1]

Z = np.zeros([len(X)-1,len(Y)-1])

for i in range(len(X)-1):
	for j in range(len(Y)-1):
		UX = np.where(abs(AR/max(AR)-X[i])<dx,1,0) # which AR values are close the X location
		SY = np.where(abs(Uf/max(Uf)-Y[j])<dy,1,0) # which U values are close to the Y location
		R = UX*SY
		Z[i,j] = np.sum(R)

x,y = np.meshgrid(X,Y)

ax = plt.figure(figsize = (9,6))
plt.pcolor(y,x,Z,shading='auto')
plt.plot(X,np.zeros([len(X),1]),'k',lw=3)
plt.plot(np.zeros([len(Y),1]),Y,'k',lw=3)
plt.text(0.125,0.25, "U > 0, AR > 0",fontsize = 18)
plt.text(-0.375,0.25, "U > 0, AR < 0",fontsize = 18)
plt.title("Advance Rate and Flow Velocity", fontsize = 18)
plt.xlabel("Advance Rate", fontsize = 18)
plt.ylabel("Flow X Velocity", fontsize = 18)


ax = plt.figure(figsize = (9,6))
plt.plot(tcut,dc[:220])
plt.plot(tcut,dt[:220])
plt.plot(tcut,Yf[:220],'k')

plt.plot(np.ones([len(t[:220]),1])*t_crit, np.linspace(max(SMA(Yf,3)),min(dt),len(t[:220])),'-k')
plt.text(.09,.00045,"ER = 0.375",fontsize = 18)
plt.text(.135,.00045,"ER = 0.40",fontsize = 18)
plt.ylabel("Vertical Distance [m]", fontsize = 18)
plt.xlabel("Time [s]", fontsize = 18)
ax.legend(["$d_c$", "$d_t$", "$Y_f$"],fontsize = 12, loc = 1)
'''
ax = plt.figure(figsize = (9,6))
plt.plot(t,Xf,'r')
plt.plot(np.ones([10,1])*t2[0],np.linspace(max(Xf),min(Xf),10),'-k')
plt.plot(np.ones([10,1])*t_crit,np.linspace(max(Xf),min(Xf),10),'k')
plt.xlabel("time [s]", fontsize = 18)
plt.ylabel("Flame X Position [m]",fontsize = 18)

plt.text(.085,.1128,"$\phi$ = 0.375",fontsize = 18)
plt.text(.135,.1128,"$\phi$ = 0.40",fontsize = 18)
plt.tick_params(direction="in")



ax = plt.figure(figsize = (9,6))
plt.plot(t,Xf,'r')
plt.plot(np.ones([10,1])*t2[0],np.linspace(max(Xf),min(Xf),10),'-k')
plt.plot(np.ones([10,1])*t_crit,np.linspace(max(Xf),min(Xf),10),'k')
plt.text(.085,.1128,"$\phi$ = 0.375",fontsize = 18)
plt.text(.135,.1128,"$\phi$ = 0.40",fontsize = 18)
plt.xlabel("time [s]", fontsize = 18)
plt.ylabel("Flame X Position [m]",fontsize = 18)
plt.tick_params(direction="in")

plt.twinx()
plt.plot(t,M2,'b')
plt.plot(t,np.ones([len(t),1]),'--b')
#plt.ylabel(r'$\frac{d_c}{d_t}$',fontsize = 18)
plt.ylabel('$Da_{FB}$', fontsize = 18)
plt.ylim([0.5,1.5])
plt.tick_params(direction="in")

plt.tick_params(direction="in")






plt.figure(figsize = (9,6))
plt.plot(t,SMA(dc2,2),'b')
plt.plot(t,SMA(dt2,2), 'k')
#plt.ylim([0.00005,16*78.125e-6/2])
plt.ylabel("Wall-Normal Distance [m]", fontsize = 18)
plt.xlabel("time [s]", fontsize = 18)
plt.legend(["$d_c$", "$d_t$"], fontsize = 18)
rectangle = plt.Rectangle((0,0), t_crit, 0.00015, fc = (0,1,0,0.5))
plt.gca().add_patch(rectangle)
rectangle = plt.Rectangle((t_crit,0), 0.17, 0.00015, fc = (1,0,0,0.5))

plt.gca().add_patch(rectangle)
plt.ylim([0,.00015])
plt.xlim([0.0152,.162])
plt.text(.10,.000025,"$\phi$ = 0.375",fontsize = 18)
plt.text(.10,.000015,"$Stable$",fontsize = 18)
plt.text(.135,.000025,"$\phi$ = 0.40",fontsize = 18)
plt.text(.135,.000015,"$Flashback$",fontsize = 18)
plt.tick_params(direction="in")
#plt.tick_params(axis='x', colors='white')
#plt.tick_params(axis='y', colors='white')
#ax.spines['bottom'].set_color('white')
#ax.spines['top'].set_color('white')
#ax.spines['right'].set_color('white')
#ax.spines['left'].set_color('white')

plt.savefig('dt_dc_t.png', transparent = True)



ax = plt.figure(figsize = (9,6))
plt.plot(t,Xf,'-r')
plt.plot(np.ones([len(t),1])*t2[0], np.linspace(max(Xf),min(Xf2),len(t)),'-k')
plt.plot(np.ones([len(t),1])*t_crit, np.linspace(max(Xf),min(Xf2),len(t)),'-k')
plt.xlabel("Time [s]", fontsize = 18)
plt.text(.085,.095,"$\phi$ = 0.375",fontsize = 18)
plt.text(.135,.095,"$\phi$ = 0.40",fontsize = 18)
plt.tick_params(direction="in")
plt.ylabel("Flame X Position [m]",fontsize = 18)
plt.twinx()
plt.plot(t,SMA(g,5),'-k')
plt.ylim([0,30e3])
plt.tick_params(direction="in")
plt.plot(t,g375*np.ones([len(t),1]),'--k')
plt.ylabel("$g$",fontsize = 18)
plt.xlabel("Time [s]", fontsize = 18)
plt.text(.08,26200,"$g_{FB}$", fontsize = 18)
ax.legend(["Flame X Position", "_", "$g$"],fontsize = 12, loc = 1)



fig, (ax0,ax1) = plt.subplots(2,1)
fig.set_size_inches((9,6))
plt.tick_params(direction="in")
ax0.plot(t,Xf,'r')
ax0.legend(["Flame X Position"], fontsize = 18)
ax1.plot(t,SMA(g,5),'-k')
#ax1.set_ylim([0,30e3])
ax1.plot(t,g375*np.ones([len(t),1]),'--k')
plt.text(.08,26200,"$g_{FB}$", fontsize = 18)
ax1.legend(["$g_c$"], fontsize = 18)












plt.show()