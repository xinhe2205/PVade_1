import numpy as np

n_panels = 10

x_all = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
T_all = np.array([100,200,300,400,500,600,700,800,900,1000])

r_out = 0.2
r_in = 0.18

G = 1e8
I_p = np.pi/2*(r_out**4-r_in**4)

theta_all = []

for i in range(np.shape(x_all)[0]):
    comp = 0
    for j in range(i):
        comp = comp+T_all[j]/G/I_p*x_all[j]
    for k in range(i, np.shape(x_all)[0]):
        comp = comp+T_all[k]/G/I_p*x_all[i]
    theta_all.append(comp)

print(theta_all)



stow_angle = 30.0/180.0*np.pi
l_0 = 0.1
d = 0.5
K = []
for i in range(np.shape(x_all)[0]):
    K.append(2*T_all[i]/l_0/l_0/np.cos(stow_angle+theta_all[i])/(np.sin(stow_angle+theta_all[i])-np.sin(stow_angle))/d)

F = []
for i in range(np.shape(x_all)[0]):
    uy = l_0/2*(np.sin(stow_angle+theta_all[i])-np.sin(stow_angle))
    F.append(K[i]*uy)

print(F)