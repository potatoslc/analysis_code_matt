'''
create a PVD file format with timesteps for all VTP files stored in that spot


'''
import os
import numpy as np

pltPATH = "/media/u1474458/Data/phi375_amr3_plt/plt/"
VTPpath = "/media/u1474458/Data/VTPs/"
prefix = "CH2_0.8"



#----------------- extract timestep array -------------
time = []
plts = os.listdir(pltPATH)

# remove all non-vtp elements in the VTP file list
for i in range(len(plts)-1,-1,-1):
	if 'plt' not in plts[i]:
		del plts[i]
plts = sorted(plts)

for i in range(len(plts)):
	if i == 85:
		continue
	f = open(pltPATH + plts[i] + '/Header')
	for j in range(36):
		f.readline()
	time.append(f.readline().strip('\n'))
	f.close()

print(time)

vtps = os.listdir(VTPpath)

# remove all non-vtp elements in the VTP file list
for i in range(len(vtps)-1,-1,-1):
	if prefix not in vtps[i] or 'pvd' in vtps[i]:
		del vtps[i]
	vtps[i].replace('.vtp', '')
	Q = vtps[i].replace('.vtp','')
	Q = Q.replace(prefix + '_','')
	vtps[i] = Q

vtps = np.array(vtps)
vtps = vtps.astype('int')	
vtps = np.sort(vtps,axis=0)


vtps2 = []
for i in range(len(vtps)):
	vtps2.append( prefix + '_' + str(vtps[i]) + '.vtp' )
#vtps = sorted(vtps)
vtps = vtps2
print(vtps)

print(len(time))
print(len(vtps))
# ------------------------ write the header -----------------------

f = open(VTPpath+prefix+'.pvd','w')
f.write('<?xml version="1.0"?>' + '\n')
f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">'  + '\n')
f.write('<Collection>' + '\n')


#------------------------------------------------------------------



for i in range(len(vtps)):
	f.write('<DataSet timestep ='+'"'+str(time[i])+'" '+ 'group=""' + ' part="0"')
	f.write(' file="' + VTPpath + vtps[i] +'"'+ '/>' + '\n')
f.write('</Collection>' + '\n')
f.write('</VTKFile>')
f.close()

