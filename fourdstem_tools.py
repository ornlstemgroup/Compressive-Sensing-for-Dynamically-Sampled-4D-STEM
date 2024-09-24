# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:48:52 2024
@author: Jacob Smith, Jordan Hatchel, and Michael Zachman - See specific function for authorship
"""
import numpy as np
from scipy.optimize import minimize

def GetCoM(f4d, cx, cy, cal, ri, ro):
    """
    Author: Jacob Smith adapted from Michael Zackman and Jordan Hachtel
    Inputs:
        f4d - a .npy or .hspy 4dstem array
        cal - the ratio for mrad per pixel
        ro - the maximum radius in pixels to consider when finding the CoM
        ri - the minimum radius in pixels to consider when finding the CoM
        cx, cy - the assumed center of the pacbed
    Output:
        sx, sy - the calculated mean CoM of the pacbed
    """
    
    X = f4d.shape[3]
    Y = f4d.shape[2]
    xran = np.linspace(int(-X/2), int(X/2), X)
    yran = np.linspace(int(-Y/2), int(Y/2), Y)
    x,y = np.meshgrid(xran*cal,yran*cal) 
    
    mask = genMask(f4d, cx, cy, cal, ri, ro)
    
    f4d = f4d*mask
    
    f4d_sum = np.sum(f4d,axis=(2,3))
    
    sx = np.sum(f4d*x,axis=(2,3))/f4d_sum
    sy = np.sum(f4d*y,axis=(2,3))/f4d_sum
    
    return sx, sy, mask

def genMask(f4d, cx, cy, cal, ri = 0, ro = np.inf):
    """
    Author: Jacob Smith
    """
    X = f4d.shape[3]
    Y = f4d.shape[2]
    
    xran = np.linspace(int(-X/2), int(X/2), X)
    yran = np.linspace(int(-Y/2), int(Y/2), Y)
    
    mask = np.ones((Y,X))
    
    # sometimes the pixels are binned into rectangles
    bin_ratio = X/Y
    
    # adjust ro and ri for per pixels
    ro = ro/cal
    ri = ri/cal
    
    if bin_ratio > 1:
        x,y = np.meshgrid(xran,yran)
        
        d = np.sqrt((x-cx+(X-1)/2)**2+((y-cy+(Y-1)/2)*bin_ratio)**2)
        
        # outer limit
        mask = mask * (d<=ro)
        # inner limit
        mask = mask * (d>=ri**2)
    elif bin_ratio < 1:
        x,y = np.meshgrid(xran,yran)
        
        d = np.sqrt(((x-cx+(X-1)/2)/bin_ratio)**2+(y-cy+(Y-1)/2)**2)
        
        # outer limit
        mask = mask * (d<=ro)
        # inner limit
        mask = mask * (d>=ri)
    elif bin_ratio == 1:
        x,y = np.meshgrid(xran,yran)
        
        d = np.sqrt((x-cx+(X-1)/2)**2+(y-cy+(Y-1)/2)**2)
    
        # outer limit
        mask = mask * (d<=ro)
        # inner limit
        mask = mask * (d>=ri)
    
    return mask

def GetVDF(f4d, VDF_def, cal, cx=0., cy=0.):
    """
    Author: Jacob Smith adapted from Michael Zackman and Jordan Hachtel
    Inputs:
        f4d - a .npy 4dstem array
        VDF_def - an Nx1 tuple containing the desired VDF detector ranges as 1x2 arrays
        cal - the ratio for mrad per pixel
        cx, cy - the center of the bright field
    Outputs:
        VDF_stack - an array stack containing the desired VDF images
    """

    VDF_stack = np.zeros((len(VDF_def),f4d.shape[0],f4d.shape[1]),dtype = 'float64')
            
    for n in range(len(VDF_def)):
        [ri,ro] = VDF_def[n]
        
        X=f4d.shape[3]
        Y=f4d.shape[2]
        
        if np.equal(cx,0.) & np.equal(cy,0.): 
            print('CoM assumed to be center of detector')
            cx,cy=X/2.,Y/2.
            
        mask = genMask(f4d, cx, cy, cal, ri, ro)
        
        VDF_stack[n,:,:] = np.sum(f4d*mask,axis=(2,3))

    return VDF_stack, mask

def CURLintensity(theta, CoM0relx, CoM0rely):
    ### Adapted by Jacob Smith from original function Written by Michael Zachman.
    # Get "Curl"

    def CURL_fun(theta):
        CoMRX,CoMRY = RotateClock(CoM0relx, CoM0rely, theta - 90)
        # Gradient(E-Field)
        CURL = (np.gradient(CoMRX)[1] + np.gradient(CoMRY)[0])
        return np.sum(np.abs(CURL))
    
    variables = minimize(CURL_fun, theta, method = 'Nelder-Mead')
    
    return variables

def RotateClock(xx,yy,theta):
    ### Written by Jordan Hachtel. Modified to use degrees by Michael Zachman.
    ### Rotate x and y 2d arrays (such as CoM or meshgrids) in the  clockwise direction by angle theta.
    ### Needed for computation of any DPC dataset with a non-zero scan rotation
    ### Needed Inputs: xx - x-coordinates of mesh grid (2d numpy array)
    ###                yy - y-coordinates of mesh grid (2d numpy array)
    ###                theta - angle for rotation (radians)    
    ### Outputs:       rx - rotated x-coordinates of meshgrid (2d numpy array)
    ###                ry - rotated y-coordinates of meshgrid (2d numpy array)
    import numpy as np
    radtheta = theta*np.pi/180
    return xx*np.cos(radtheta)-yy*np.sin(radtheta),xx*np.sin(radtheta)+yy*np.cos(radtheta)

