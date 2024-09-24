# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:07:07 2024



Authors:
Principle machine learning model and supporting code developed by Hoang Tran, Zhaiming Shen
Principle 4D-STEM interpretation code developed by Jacob Smith

"""

#%% Import and Setup Libraries

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import os
import cv2
from torch import nn
import fourdstem_tools as fourT
import random

from torch.utils.data import Dataset, DataLoader

from utils.nt_toolbox.general import *
from utils.nt_toolbox.signal import *
from utils.nt_toolbox.perform_wavelet_transf import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device: %s'%device)

#%% User input parameters

num_epochs = 1
batch_size = 32

#%%
#------------------
# ML models
#------------------

class Data(Dataset):
    def __init__(self, sample_size, is_train):
        
        if is_train == True:
            self.xx = torch.tensor(data_sample)            
            self.yy = torch.tensor(X_train)
            self.len=self.xx.shape[0]
        else:
            # self.xx = torch.tensor(data_unsample)
            # self.yy = torch.tensor(X_test)
            
            # load both training and test sets
            self.xx = torch.tensor(data_total)
            self.yy = torch.tensor(X)
            self.len=self.xx.shape[0]
            
    def __getitem__(self,index):

        return self.xx[index].float(),self.yy[index].float()
    
    def __len__(self):
        return self.len
    
    
class LAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()     

        ### encoder cnn layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.Conv2d(4, 8, 3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            )
       

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 3, padding=0, stride=2, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 4, 3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4, 1, 3, padding=1, stride=2, output_padding=1),
            nn.Tanh()
            )   

        self.fc1=nn.Linear(5,10000)
        # self.fc2=nn.Linear(20,80)
        # self.fc3=nn.Linear(80,250)
        # self.fc4=nn.Linear(250,800)
        self.fc5=nn.Linear(10000,2048)

        
    def forward_ae(self, x):
        

        latent = self.encoder(x)
        x = self.decoder(latent)

        return x, latent

    def forward_fc(self,x):
        
        yhat = torch.relu(self.fc1(x))
        # yhat = torch.relu(self.fc2(yhat))
        # yhat = torch.relu(self.fc3(yhat))
        # yhat = torch.relu(self.fc4(yhat))
        yhat = self.fc5(yhat)
        
        return yhat


    def reconst(self, x):
        #x = self.decoder_lin(x)
        #x = self.unflatten(x)
        x = self.decoder(x)

        return x 
    
#%%

def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    xx = x * (ub - lb) + lb
    return xx 


def to_unit_cube(xx, lb, ub):
    """Project from the hypercube with bounds lb and ub to [0, 1]^d"""
    x = (xx - lb)/(ub - lb)  
    return x


def integ_img(f4d): 

    '''generate integrated image from 4D data '''

    bfr = 34      # Central (bright field) disk radius in pixels
    alpha = 28     # Semiconvergence angle in mrad
    rin = 0     # Inner angle in pixels, as an option to be used in masking CoM calculation
    rout = np.inf     # Outer angle in pixels, as an option to be used in masking CoM calculation

    diffcal=alpha/bfr

    RonchiMean=np.mean(f4d,axis=(0,1))
    meanCoM = fourT.GetCoM(f4d, RonchiMean.shape[1]/2, RonchiMean.shape[0]/2, diffcal, rin, rout)
    meanCoMx = np.mean(meanCoM[0])
    meanCoMy = np.mean(meanCoM[1])
    cx = meanCoMx + (f4d.shape[3] - 1)/2
    cy = meanCoMy + (f4d.shape[2] - 1)/2

    # BF, ABF and ADF

    BFin = 0.0        # Inner angle for BF image in mrad
    BFout = 20.0       # Outer angle for BF image in mrad
    ABFin = 5.0       # Inner angle for ABF image in mrad
    ABFout = 30.0      # Outer angle for ABF image in mrad
    ADFin = 35.0       # Inner angle for ADF image in mrad
    ADFout = np.inf    # Outer angle for ADF image in mrad

    VDF_def = ([BFin,BFout],[ABFin,ABFout],[ADFin,ADFout])

    [VDF_stack, masks] = fourT.GetVDF(f4d, VDF_def, diffcal, cx, cy)

    # 

    ri = rin     # Inner angle for calculation of relative scan-detector rotation angle, in pixels
    ro = rout     # Outer angle for calculation of relative scan-detector rotation angle, in pixels
    theta = 0

    CoM0 = fourT.GetCoM(f4d,cx,cy,diffcal,ri,ro)
    CoM0relx = -CoM0[0] + (f4d.shape[3] - 1)/2 - cx
    CoM0rely = CoM0[1] + (f4d.shape[3] - 1)/2 - cy
    ang = fourT.CURLintensity(theta, CoM0relx, CoM0rely)
    scandetang = ang.x

    # backgrgauss = 50    # Size of gaussian to use for long-range background (improper descan) removal.

    CoMX0 = CoM0relx - np.mean(CoM0relx)   # Remove zero-frequency component, from misalignment of CoM center.
    CoMY0 = CoM0rely - np.mean(CoM0rely)   # Remove zero-frequency component, from misalignment of CoM center.

    #scandetang = scandetang + 180     # Uncomment if inversion of sign of atoms is necessary.
    #CoMX0 = CoMX0 - ndimage.gaussian_filter(CoMX0, backgrgauss)   # Uncomment if long-range background needs to be removed.
    #CoMY0 = CoMY0 - ndimage.gaussian_filter(CoMY0, backgrgauss)   # Uncomment if long-range background needs to be removed.

    CoMX,CoMY = fourT.RotateClock(CoMX0, CoMY0, scandetang)

    return np.concatenate((VDF_stack,np.expand_dims(CoMX,axis=0),np.expand_dims(CoMY,axis=0)),axis = 0)


#%%
#------------------
# MAIN CODES
#------------------

#------------------
# import 4d spectrum image and reconstruct using compressed sensing
#------------------

data_original = np.load('6MEO/my_Spectrum_Image.npy')
data_name = '6MEO'

path  = 'figs/' + data_name + '_adaptive_sampl'
path2 = path + '/diffraction_examples_testing'
path3 = path + '/diffraction_examples_training'
if not os.path.exists(path): 
    os.mkdir(path)
    os.mkdir(path2)
    os.mkdir(path3)

dim = data_original.shape[0]
dim_diff = data_original.shape[2]

print(dim,dim_diff)

score_list = ['BF','ABF','ADF','CoMX','CoMY']   # list of STEM names
f_list = []   # array for CS reconstructed images

# 'BF':   img_stack[0,:,:] 
# 'ABF':  img_stack[1,:,:] 
# 'ADF':  img_stack[2,:,:]
# 'CoMX': img_stack[3,:,:]
# 'CoMY': img_stack[4,:,:]
img_stack = integ_img(data_original)

# generate subsampling mask from smoothed ADF image 
# pixels will be sampled with probability related to mask_gen values
mask_gen = img_stack[2,:,:] 
mask_gen = (mask_gen-np.min(mask_gen)) * 255 /(np.max(mask_gen) - np.min(mask_gen))
mask_gen = Image.fromarray(np.uint8(mask_gen), 'L')

mask_gen = mask_gen.filter(ImageFilter.GaussianBlur(5))

mask_gen = np.array(mask_gen)/255

plt.figure()
plt.imshow(mask_gen)
plt.colorbar()

grad_mask_gen = np.zeros_like(mask_gen)

for i in range(len(mask_gen)): 
    for j in range(len(mask_gen)): 
        if i == len(mask_gen)-1:
            grad_y = mask_gen[i-1,j] - mask_gen[i,j]
        else: 
            grad_y = mask_gen[i,j] - mask_gen[i+1,j]
        if j == len(mask_gen)-1:
            grad_x = mask_gen[i,j-1] - mask_gen[i,j]
        else: 
            grad_x = mask_gen[i,j] - mask_gen[i,j+1]
            
        grad_mask_gen[i,j] = np.sqrt(grad_x**2 + grad_y**2) 
        
mask_gen = (mask_gen + 80*grad_mask_gen)/np.max(mask_gen + 80*grad_mask_gen)

# order 2: 21% 
#       1: 40% 
#       1/2: 60% 
#       1/3: 70% 
mask_gen = mask_gen**(1)

plt.figure()
plt.imshow(mask_gen)
plt.colorbar()

Omega = np.zeros_like(mask_gen)
for i in range(len(Omega)): 
    for j in range(len(Omega)): 
        Omega[i,j] = np.random.choice(np.arange(0, 2), p=[mask_gen[i,j],1-mask_gen[i,j]])

plt.figure()     
plt.imshow(Omega,cmap='gray')

rate = (Omega==0).sum()/(dim*dim)
print('Sampling rate = %.2f' %rate)

# record unsampled index 
sel = np.argwhere(Omega==1)

# save original and subsampled images for plotting
# original01_list here is loaded from STEM image and scales in [0,1]
# difference from later original_list constructed from diffraction pattern, which have various scales
original01_list = []
subsample_list = []

# saving max and min ABF, ADF, etc. scores (for de-normalization)
max_list = []
min_list = []


for j in range(len(score_list)):
    # original STEM image
    f0 = img_stack[j,:,:]
    
    minf0 = np.min(f0) 
    maxf0 = np.max(f0)
    
    print('min %s  = %.2f' %(score_list[j],minf0))
    print('max %s  = %.2f' %(score_list[j],maxf0))
    
    f0 = (f0-minf0)/(maxf0-minf0)

    # construct subsampled STEM image (random sampling)
    Phi = lambda f, Omega: f*(1-Omega)
    y = Phi(f0, Omega)
    
    original01_list.append(f0)
    subsample_list.append(y)
    
    max_list.append(maxf0)
    min_list.append(minf0)

    # plt.figure(figsize = (6,6))
    # plt.imshow(y, cmap = 'gray')
    # plt.savefig('tmp.png',dpi=300)
    # plt.show()

    # reconstruct STEM image by hard-thresholding
    ProjC = lambda f, Omega, y: Omega*f + (1-Omega)*y
    Jmax = np.log2(dim)-1
    Jmin = (Jmax-3)

    J = Jmax-Jmin + 1
    u = np.hstack(([4**(-J)], 4**(-np.floor(np.arange(J + 2./3,1,-1./3)))))
    U = np.transpose(np.tile(u, (dim,dim,1)),(2,0,1))

    lambd = .01
    Xi = lambda a: perform_wavelet_transf(a, Jmin, -1, ti=1)
    PsiS = lambda f: perform_wavelet_transf(f, Jmin, + 1, ti=1)
    Psi = lambda a: Xi(a/U)

    tau = 1.9*np.min(u)
    HardThresh = lambda x, t: x*(abs(x) > t)

    niter = 500
    lambda_list = np.linspace(1, 0, niter)

    fHard = y
    fHard = ProjC(fHard, Omega, y)
    fHard = Xi(HardThresh(PsiS(fHard), tau*lambda_list[1]))

    for i in range(niter):
        fHard = Xi(HardThresh(PsiS(ProjC(fHard, Omega, y)), lambda_list[i]))
        
    rel_error = np.linalg.norm(clamp(fHard)-f0)/np.linalg.norm(f0)

    # de-normalize CS-reconstructed STEM image 
    fHard = clamp(fHard)*(maxf0-minf0) + minf0

    f_list.append(fHard)

    # imageplot(clamp(fHard), "Inpainting hard thresh., PSNR = %.1f dB" % psnr(f0, fHard))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
    

    for index, ax in enumerate(axes.flat):
        if index == 0:
            im = ax.imshow(f0, cmap='gray')
            ax.title.set_text('Original STEM image (%s)'%score_list[j])
        if index == 1:
            im = ax.imshow(y, cmap = 'gray')
            ax.title.set_text('Subsampled STEM image (%s). Rate = %.2f'%(score_list[j],rate))
        if index == 2:
            im = ax.imshow(f_list[j], cmap='gray')
            ax.title.set_text("CS reconstruction of STEM image (%s) \n PSNR = %.2f dB; Relative error = %.4f"\
                              % (score_list[j], psnr(f_list[j],f0), rel_error))
            
    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(path+'/CS_STEM_%s.png'%(score_list[j]), dpi =300)

    plt.show()
    
#%%
#------------------
# preprocess data for autoencoder training 
#------------------

max_list = torch.FloatTensor(max_list)
min_list = torch.FloatTensor(min_list)

# flatten  
data_total = np.zeros((dim**2, 1, dim_diff, dim_diff))
index = np.arange(dim**2)
unsample_index = np.zeros(sel.shape[0],dtype=int)

for i in range(dim):
    for j in range(dim):
        data_total[j+dim*i,:,:,:] = data_original[i,j,:,:] 

# normalize
data_max = np.max(data_total, axis = (0,1,2,3))
data_min = np.min(data_total, axis = (0,1,2,3))
data_total = (data_total - data_min)/(data_max-data_min)

# random sample
for i in range(len(unsample_index)):
    unsample_index[i] = int(sel[i,1] + dim*sel[i,0])
    
random_index = [i for i in index if i not in unsample_index]
random_index = np.array(random_index, dtype=int)

# print(set(unsample_index))
# print(set(random_index))

data_sample = data_total[random_index,:,:,:]
data_unsample = data_total[unsample_index,:,:,:]


# show an example of diffraction pattern
random_choice = 1123
plt.imshow(data_total[random_choice, 0, :, :])
plt.colorbar()
plt.show()


#------------------
# load 2d STEM image
#------------------

X1 = np.reshape(f_list[0], (dim**2,1))
X2 = np.reshape(f_list[1], (dim**2,1))
X3 = np.reshape(f_list[2], (dim**2,1))
X4 = np.reshape(f_list[3], (dim**2,1))
X5 = np.reshape(f_list[4], (dim**2,1))

X = np.concatenate((X1, X2, X3, X4, X5), axis=1)

#subsample the data
X_train = X[random_index, :]
X_test = X[unsample_index, :]

# plt.imshow(f2, cmap="gray")

#------------------
# training 
#------------------
        
train_sample_size = data_sample.shape[0]

print('training sample size =', train_sample_size)

data_set=Data(train_sample_size, True)  # input data_sample and X_train 
data_loader=DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

index_min = 33
index_max = 99


weights = [1, 1, 1, 1]

learning_rate = 5e-4

model_AE = LAutoencoder()
model_AE.to(device)
# criterion_ae = nn.MSELoss()
criterion_ae_in = nn.MSELoss()
criterion_ae_out_h = nn.MSELoss()
criterion_ae_out_v = nn.MSELoss()
criterion_fc = nn.MSELoss()
optimizer = torch.optim.AdamW(model_AE.parameters(), lr=learning_rate, weight_decay = 1e-8)
# optimizer = torch.optim.Adam(model_AE.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08,weight_decay = 1e-8)

for epoch in range(num_epochs):
    for data, x in data_loader:
        data = data.to(device)
        x = to_unit_cube(x, min_list, max_list).float()
        x = x.to(device)
        output, latent_sample = model_AE.forward_ae(data.float())
        yhat = model_AE.forward_fc(x)
        output_in = output[:,:,index_min:index_max,index_min:index_max]
        data_in = data[:,:,index_min:index_max,index_min:index_max]
        output_out_h = output[:,:,np.r_[:index_min,index_max:],:]
        data_out_h = data[:,:,np.r_[:index_min,index_max:],:]
        output_out_v = output[:,:,:,np.r_[:index_min,index_max:]]
        data_out_v = data[:,:,:,np.r_[:index_min,index_max:]]

        # loss_ae = criterion_ae(output.float(), data.float())
        loss_ae_in = criterion_ae_in(output_in.float(), data_in.float())
        loss_ae_out_h = criterion_ae_out_h(output_out_h.float(), data_out_h.float())
        loss_ae_out_v = criterion_ae_out_v(output_out_v.float(), data_out_v.float())

        latent_sample = torch.reshape(latent_sample, (latent_sample.size()[0],-1))

        #pca = PCA(n_components = 200) 
        #latent_sample_reduced = pca.fit_transform(latent_sample.cpu().detach().numpy())  
        #yhat_reduced = pca.fit_transform(yhat.cpu().detach().numpy())  

        # [U, S, V] = torch.pca_lowrank(latent_sample, 256)
        # latent_sample_reduced = torch.matmul(latent_sample, V[:, :256])
        # [U, S, V] = torch.pca_lowrank(yhat, 256)
        # yhat_reduced = torch.matmul(yhat, V[:, :256])

        loss_fc = criterion_fc(yhat, latent_sample)
        loss = (weights[0]*loss_ae_in + weights[1]*loss_ae_out_h + weights[2]*loss_ae_out_v + weights[3]*loss_fc)/sum(weights)
        # loss = loss_ae + loss_fc
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.data.item()}, loss_ae:{loss_ae.data.item()}, loss_fc:{loss_fc.data.item()}')
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.data.item()}, loss_ae_in:{loss_ae_in.data.item()}, loss_ae_out_h:{loss_ae_out_h.data.item()}, loss_ae_out_v:{loss_ae_out_v.data.item()}, loss_fc:{loss_fc.data.item()}')

#%%
#------------------------
# Reconstruct STEM images from predicted 4d diffraction 
#------------------------

test_sample_size = data_total.shape[0]

# re-define batch size
batch_size   = 512
batch_number = test_sample_size//batch_size if test_sample_size % batch_size == 0 else test_sample_size//batch_size + 1

for i in range(batch_number):
    if (i+1)%50 == 0:
        print('evaluate [%d/%d]'%(i+1,batch_number))

    if i<batch_number-1: 
        data = torch.tensor(data_total[i*batch_size:(i+1)*batch_size,:,:,:]).float().to(device)
        x    = torch.tensor(X[i*batch_size:(i+1)*batch_size,:]).cpu().detach()
        x    = to_unit_cube(x, min_list, max_list).float().to(device)

        
    else: 
        data = torch.tensor(data_total[i*batch_size:,:,:,:]).float().to(device)
        x    = torch.tensor(X[i*batch_size:,:]).cpu().detach()
        x    = to_unit_cube(x, min_list, max_list).float().to(device)


    output_now, _ = model_AE.forward_ae(data)
    yhat_now      = model_AE.forward_fc(x)
    pred_4d_now   = model_AE.reconst(torch.reshape(yhat_now,(batch_size,32,8,8)))

    if i == 0: 
        output  = output_now.cpu().detach().numpy()
        yhat    = yhat_now.cpu().detach().numpy()
        pred_4d = pred_4d_now.cpu().detach().numpy()
    else: 
        output  = np.concatenate((output,output_now.cpu().detach().numpy()),axis=0)
        yhat    = np.concatenate((yhat,yhat_now.cpu().detach().numpy()),axis=0)
        pred_4d = np.concatenate((pred_4d,pred_4d_now.cpu().detach().numpy()),axis=0)

yhat = yhat.reshape((test_sample_size, 32, 8, 8))

# de-normalize reconstructed 4d images
data_reconstr = pred_4d.reshape((dim,dim,dim_diff,dim_diff)) * (data_max-data_min) + data_min

img_reconstr_stack = integ_img(data_reconstr)

print("BF relative Error:", np.linalg.norm(img_reconstr_stack[0,:,:] - img_stack[0,:,:])/np.linalg.norm(img_stack[0,:,:]))
print("ABF relative Error:", np.linalg.norm(img_reconstr_stack[1,:,:] - img_stack[1,:,:])/np.linalg.norm(img_stack[1,:,:]))
print("ADF relative Error:", np.linalg.norm(img_reconstr_stack[2,:,:] - img_stack[2,:,:])/np.linalg.norm(img_stack[2,:,:]))
print("CoMX relative Error:", np.linalg.norm(img_reconstr_stack[3,:,:] - img_stack[3,:,:])/np.linalg.norm(img_stack[3,:,:]))
print("CoMY relative Error:", np.linalg.norm(img_reconstr_stack[4,:,:] - img_stack[4,:,:])/np.linalg.norm(img_stack[4,:,:]))

# # save reconstructed 4D data
# np.save('reconstruct_Spectrum_Image_rate%.2f.npy'%rate, np.transpose(data_reconstr,(1,0,2,3)))

#%%   
# plot and save true and approximate STEM images 

for j in range(len(score_list)):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
    
    rel_error = np.linalg.norm(img_reconstr_stack[j,:,:] - img_stack[j,:,:])/np.linalg.norm(img_stack[j,:,:])

    for index, ax in enumerate(axes.flat):
        if index == 0:
            im = ax.imshow(img_stack[j,:,:], cmap='gray')
            ax.title.set_text('Original STEM image (%s)'%score_list[j])
        if index == 1:
            im = ax.imshow(subsample_list[j], cmap = 'gray')
            ax.title.set_text('Subsampled STEM image (%s). Rate = %.2f'%(score_list[j],rate))
        if index == 2:
            im = ax.imshow(img_reconstr_stack[j,:,:], cmap='gray')
            ax.title.set_text("Reconstructed STEM image (%s) from AE approximation of diffraction \n PSNR = %.2f dB; Relative error = %.4f"\
                              % (score_list[j], psnr(img_reconstr_stack[j,:,:],img_stack[j,:,:]), rel_error))

    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(path+'/AE_STEM_%s.png'%(score_list[j]), dpi =300)

    plt.show()
    
# extract diffraction pattern at some random (UNSAMPLED) locations
random_idx = unsample_index[np.random.randint(0,len(unsample_index), 400) ]

for i in range(len(random_idx)): 
    
    k = random_idx[i]
    
    RGB_orig = cv2.cvtColor(np.float32(original01_list[0]),cv2.COLOR_GRAY2BGR)
    
    # create marker for the location on STEM image 
    loc = []
    center = np.zeros(2)
    center[0] = k // dim
    center[1] = k %  dim
    loc.append(center)
    if center[1]>0:        loc.append(center - [0,1]) 
    if center[0]>0:        loc.append(center - [1,0]) 
    if center[1]<dim-1:      loc.append(center + [0,1]) 
    if center[0]<dim-1:      loc.append(center + [1,0]) 
    if center[1]>0 and center[0]>0:     loc.append(center - [1,1]) 
    if center[1]>0 and center[0]<dim-1:   loc.append(center - [-1,1]) 
    if center[1]<dim-1 and center[0]>0:   loc.append(center - [1,-1]) 
    if center[1]<dim-1 and center[0]<dim-1: loc.append(center - [-1,-1]) 
    
    for j in range(len(loc)): 
        RGB_orig[int(loc[j][0]), int(loc[j][1])] = 1,0,0
                        
    error1 = np.linalg.norm(data_total[k,0,:,:] - output[k,0,:,:])/np.linalg.norm(data_total[k,0,:,:])
    error2 = np.linalg.norm(data_total[k,0,:,:] - pred_4d[k,0,:,:])/np.linalg.norm(data_total[k,0,:,:])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28,7))
    
    for index, ax in enumerate(axes.flat):
        if index == 0:
            im = ax.imshow(RGB_orig)
            ax.title.set_text('Location in STEM image')
        if index == 1: 
            im = ax.imshow(data_total[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('Normalized original Diffraction Pattern')
        if index == 2:
            im = ax.imshow(output[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('(Normalized reconstructed Diffraction Pattern via Autoencoder. \n Relative Error: ' + str(error1))
        if index == 3:
            im = ax.imshow(pred_4d[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('Normalized reconstructed Diffraction Pattern from STEM data via FC and Decoder. \n Relative Error: ' + str(error2))

    plt.savefig(path2 + '/diff_%d_%d.png'%(center[0],center[1]), dpi =300)
    
# extract diffraction pattern at some random (SAMPLED) locations
random_idx2 = random_index[np.random.randint(0,len(random_index), 400) ]

for i in range(len(random_idx2)): 
    
    k = random_idx2[i]
    
    RGB_orig = cv2.cvtColor(original01_list[0],cv2.COLOR_GRAY2BGR)
    
    # create marker for the location on STEM image 
    loc = []
    center = np.zeros(2)
    center[0] = k // dim
    center[1] = k %  dim
    loc.append(center)
    if center[1]>0:        loc.append(center - [0,1]) 
    if center[0]>0:        loc.append(center - [1,0]) 
    if center[1]<dim-1:      loc.append(center + [0,1]) 
    if center[0]<dim-1:      loc.append(center + [1,0]) 
    if center[1]>0 and center[0]>0:     loc.append(center - [1,1]) 
    if center[1]>0 and center[0]<dim-1:   loc.append(center - [-1,1]) 
    if center[1]<dim-1 and center[0]>0:   loc.append(center - [1,-1]) 
    if center[1]<dim-1 and center[0]<dim-1: loc.append(center - [-1,-1]) 
    
    for j in range(len(loc)): 
        RGB_orig[int(loc[j][0]), int(loc[j][1])] = 1,0,0
                        
    error1 = np.linalg.norm(data_total[k,0,:,:] - output[k,0,:,:])/np.linalg.norm(data_total[k,0,:,:])
    error2 = np.linalg.norm(data_total[k,0,:,:] - pred_4d[k,0,:,:])/np.linalg.norm(data_total[k,0,:,:])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28,7))
    
    for index, ax in enumerate(axes.flat):
        if index == 0:
            im = ax.imshow(RGB_orig)
            ax.title.set_text('Location in STEM image')
        if index == 1: 
            im = ax.imshow(data_total[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('Normalized original Diffraction Pattern')
        if index == 2:
            im = ax.imshow(output[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('(Normalized reconstructed Diffraction Pattern via Autoencoder. \n Relative Error: ' + str(error1))
        if index == 3:
            im = ax.imshow(pred_4d[k,0,:,:], vmin=0, vmax=1)
            ax.title.set_text('Normalized reconstructed Diffraction Pattern from STEM data via FC and Decoder. \n Relative Error: ' + str(error2))

    plt.savefig(path3 + '/diff_%d_%d.png'%(center[0],center[1]), dpi =300)
    





