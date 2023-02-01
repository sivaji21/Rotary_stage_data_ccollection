# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:14:25 2022.

@author: s344542
"""
#%% 
"""
setup
"""

import ensemble_stages
import numpy as np
import time
from datetime import datetime
import pandas as pd
import rri_data_puller
import pyximea as xi #camera interface module
import matplotlib.pyplot as plt
import pycorrelators as pycor
import socket_controller3 as sc
controller = sc.SocketController()
controller.start_in_thread()
plt.ion()


pycor.load_fft_planning('C:\\Users\\s344542\OneDrive - Cranfield University\\python_modules\\fftplans')

cam = xi.Camera(0)
cam.set_fps(60) # setting frames per second
cam.set_exposure_time(2500) # setting camera exposure
rri = rri_data_puller.RRI_data_logger(controller, host = '192.168.1.5',  port = 6801)


stage = ensemble_stages.ThetaStage()

# stage.set_acceleration(1)
ang_temp =[] #input angles

#%%
"""
Taking a snap
"""


img,info = cam.snap() # take a snap img stores the pic as a matrix info stores the metadata                   
plt.imshow(img)
plt.colorbar()
plt.savefig('img',dpi=400) #save the image
img0 = img

#%%


"""
setting a reference for correlation
 
"""

img0 = img 
i0,j0 = 422,387  # center of target(y,x from centre)
w,h = 160,160     # width, height to crop
t = img0[i0-h//2: i0+h//2+1,j0-w//2: j0+w//2+1 ] # t is the target to search
t = np.ascontiguousarray(t)
plt.figure()
plt.imshow(t)
#%%
"""
not necessary move to bottom
"""
img0 = img
cor = pycor.nxcorr2(fshape=img0.shape, tshape=t.shape)
pycor.save_fft_planning('C:\\Users\\s344542\OneDrive - Cranfield University\\python_modules\\fftplans')
#%% 

"""
measurement cell
10 degrees
"""
rri.reset_data()
step = .3 # step size in degrees
max_angle = 10
vel  = 20  # speed to move stages between steps deg/s
wait = 2 # pause at each step
stage.set_acceleration(20)
#angles to step through
angs = np.hstack( (np.arange(0,max_angle + step, step))  )


# angs_lst=[0,5,10,15,20,25,30,35,40]
# angs=np.array(angs_lst)

"""
#start rri measurement
click the start button in rri box
"""
#step through angles
read_angs = []
stage_times =[]
image_captured=[]

stage.move_abs(-1, vel, wait=True)
stage.move_abs(0, vel, wait=True)
for ang in angs:
    stage_time = time.time()
    stage_times.append(stage_time)
    stage.move_abs(ang, vel, wait=True)
    img,info = cam.snap() # take a snap img stores the pic as a matrix info stores the metadata  
    read_ang = stage.position
    read_angs.append(read_ang)
    image_captured.append(img)
    time.sleep(wait)
      
#%%
"""
#get the rri data and stage data
"""
t_rri,phase,amp = rri.get_data()
phase = phase.T
time_start = t_rri[0] # first reading of time
time_c_rri = t_rri-time_start # subtracr first reading from all other time readings to start from 0
rri_combined = np.hstack((time_c_rri.reshape(-1,1),phase))

"""
stage data save to npz 1,2,3
"""
s_read_angle  = pd.DataFrame(read_angs) # save to npz 1
s_t = pd.DataFrame(stage_times) 
s_t_corrected = s_t - stage_times[0]# save to npz 2
# stage_read_input = np.hstack((s_t_c,s,angs.reshape(-1,1)))
#input_ang =np.hstack((s_t_c,angs.reshape(-1,1))) # save to npz 3
s_input_ang =np.hstack(angs.reshape(-1,1))

fl = datetime.now().strftime("%d_%m_%H_%M_%S")







#%%
"""
corelation image
"""
 
cc = cor(img0, t)
plt.colorbar()
plt.imshow(cc)

#%%
"""
finding xy_final_list
"""
xy_final_list=[]

data1=image_captured#np.load('images_24_01_16_09_49.npy')

#get a template from first image (0 degrees)
i0,j0 = 422,387  # center of target(y,x from centre)
w,h = 160,160     # width, height to crop
t = img0[i0-h//2: i0+h//2+1,j0-w//2: j0+w//2+1 ] # t is the target to search
t = np.ascontiguousarray(t)
plt.figure()
plt.imshow(t)

cor = pycor.nxcorr2(fshape=img0.shape, tshape=t.shape)
pycor.save_fft_planning('C:\\Users\\s344542\OneDrive - Cranfield University\\python_modules\\fftplans')

for img in data1:
    
    cc = cor(img, t)
    y0,x0,q0 = pycor.find_peak2d_gaussian9p(cc)
    y0 = y0-t.shape[0]//2
    x0 = x0-t.shape[1]//2
    
    xy_final_list.append( (x0,y0) )
np.save( './xy_final_list_'+fl+'.npy', xy_final_list)
    
    
    
    
#%%
"""
finding angle from xy finalxy_final_list
"""
xy=np.array([xy_final_list]) # coverting list to np.array 
                            #xy.shape Out[61]: (1, 35, 2)
                            
dx = xy[0,:,0]
dy = xy[0,:,1]
scale =0.03337 # mm/pix
r = 99.3        #radius mm measured using calipers
dx_mm = (dx-dx[0])*scale
dy_mm = (dy-dy[0])*scale
def calculate_angles(dx_mm, dy_mm, scale =0.03337,r =99.3 ):
    """
    Takes target x,y image coordinates, scales to mm using the scale factor and
    removes the offset (i.e. first point is 0,0). 
    Then converts to angles by calculating angle shift between each then
    integrating to find angle change from first point.
    """


    #check error in simple angle calulation
    dxs = np.diff(dx_mm)
    dys = np.diff(dy_mm)
    dss = np.hypot(dxs, dys)

    #step in angle between each point
    dts = np.degrees( np.arctan2(dss,r))

    #summed up shift in angle 
    ts = np.cumsum(dts)
    
    #add the zero angle for first point
    ts = np.hstack([0,ts])
    
    return ts
"""
save to npz 4
"""
angles_ximea = calculate_angles(dx_mm,dy_mm,scale,r)
#%%
"""
save the data
planning to save as .npz file
"""

rri_time =rri_combined[:,0]
A1 =rri_combined[:,0]
A2 =rri_combined[:,0]
A3 =rri_combined[:,0]
B1 =rri_combined[:,0]
B2 =rri_combined[:,0]
B3 =rri_combined[:,0]

"""
COVERT TO SEGMENTS
"""
SA10 = A1        #tip sensor
SA21 = A2-A1     #joint 4 working on this
SA32 = A3-A2    #joint 3



SB10 = B1        #tip sensor
SB21 = B2-B1     #joint 4
SB32 = B3-B2    #joint 3

"""
finding angle from RRI

"""
wl0 = 1550e-9            #wavelength
delta = 0.000580000000000#600e-6          #fibre spacing(.53mm)
ng = 1.4682             #group refractive index
k =  0.79                   #fibre strain sensitivty factor (from tk paper)
"""
save to npz 5
"""
ang2 = np.degrees(-(SA21-SB21)*wl0 / (delta*4*np.pi*ng*k))

"""
saving as npz
"""
fl = datetime.now().strftime("%d_%m_%H_%M_%S")
np.savez( './combined_'+fl+'_angles', ang2=ang2,rri_time=rri_time,s_read_angle=s_read_angle,
         s_t_corrected=s_t_corrected,s_input_ang=s_input_ang,angles_ximea=angles_ximea)

"""
ideas for graph
auto load latest .npz
list each with their respective name e.g:print(d['a'])
moving the .npz to a folder with date time

"""
# SB43 = B4-B3   #joint 2
#np.savetxt( './input_ang_'+fl+'.csv', input_ang, delimiter=',',header="time,Angle", comments='')
# np.savetxt( './stage_angle'+fl+'.csv', stage_read_input, delimiter=',',header="time,Read,input", comments='')
#np.save( './images_'+fl+'.npy', image_captured)
# np.savetxt( './rri_'+fl+'.csv', rri_combined, delimiter=',',header="t,a1,a2,a3,b1,b2,b3", comments='')
# np.save( './all_'+fl+'.npy', image_captured,rri_combined,stage_read_input,s_t_c)
#%%

"""
move  data to folder
"""
import os
import shutil
from datetime import datetime



source = r'C:/Users/s344542/OneDrive - Cranfield University/code_written_by_day/31_01_23'
destination = datetime.now().strftime("%d_%m_%H_%M_%S")
"""
move  data to folder
"""
def move(): # defining the statement move

    if not os.path.exists(source+datetime.now().strftime("%d_%m_%H_%M_%S")): #if folder called im is not present
      os.makedirs(datetime.now().strftime("%d_%m_%H_%M_%S")) #create a folder datetime
      for i in os.listdir(source): # list all files in source folder
          if i.endswith('.npz'): #if it is a .csv file
              shutil.move(os.path.join(source,i),destination) #move the file to destination


move()
#%%

"""
finding peak x,y co ordinates
"""
y0,x0,q0 = pycor.find_peak2d_gaussian9p(cc)
s=round(y0) # value rounded
f=x0
print(s)
plt.figure()
plt.plot(cc[s,:])
mcc = np.abs(cc) # Find magnitude
peak_x_cc = np.max(mcc) # Find max peak
print(peak_x_cc,f)


