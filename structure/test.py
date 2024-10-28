import numpy as np
from scipy.interpolate import griddata

top_coor_load = np.loadtxt('top_surface_load.txt')
bot_coor_load = np.loadtxt('bot_surface_load.txt')

aa = top_coor_load[:,1]/24.2*20.0
bb = top_coor_load[:,0]/4.1*4.0

top_coor_load[:,0] = aa
top_coor_load[:,1] = bb

aaa = bot_coor_load[:,1]/24.2*20.0
bbb = bot_coor_load[:,0]/4.1*4.0

bot_coor_load[:,0] = aaa
bot_coor_load[:,1] = bbb

top_coor = top_coor_load[:,:3]
bot_coor = bot_coor_load[:,:3]


print(np.min(top_coor_load[:,0]), np.max(top_coor_load[:,0]))
print(np.min(top_coor_load[:,1]), np.max(top_coor_load[:,1]))
print(np.min(top_coor_load[:,2]), np.max(top_coor_load[:,2]))

print(np.min(bot_coor_load[:,0]), np.max(bot_coor_load[:,0]))
print(np.min(bot_coor_load[:,1]), np.max(bot_coor_load[:,1]))
print(np.min(bot_coor_load[:,2]), np.max(bot_coor_load[:,2]))

aaa = np.array([1,2,3,4])
bbb = np.array([11,22,33,44])
print(aaa.T)


ccc = np.zeros((4,2))
ccc[:,0] = aaa
ccc[:,1] = bbb
print(ccc)