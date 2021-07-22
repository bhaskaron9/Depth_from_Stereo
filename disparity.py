import cv2
import numpy as np
from util import NCC
def find_disp_NCC(imgL, imgR, win=5, disp_rng=16):
  # find the disparity of left image with NCC
  img_dispL = np.zeros(imgL.shape).astype(np.float64)
  sz = win//2
  imgL_I = cv2.copyMakeBorder(imgL, sz, sz, sz, sz, cv2.BORDER_REFLECT)
  imgR_I = cv2.copyMakeBorder(imgR, sz, sz, sz, sz, cv2.BORDER_REFLECT)
  for i in range(sz, imgL.shape[0]+sz):
    for j in range(sz, imgL.shape[1]+sz):
      patchL = imgL_I[i-sz:i+sz+1,j-sz:j+sz+1]
      max_ncc = 0
      for k in range(max(sz, j-disp_rng), min(j+disp_rng, imgR.shape[1]+sz)):
        patchR = imgR_I[i-sz:i+sz+1, k-sz:k+sz+1]
        ncc = NCC(patchL, patchR)
        if ncc>max_ncc:
          max_ncc = ncc
          img_dispL[i-sz, j-sz] = abs(k-j)
  return img_dispL  

def find_disp_DP(imgL, imgR, occlusion_cost = 0.01, win=3,  max_disp = 16):
  # find the disparity of left and right image with DP
  
  img_dispL = np.zeros(imgL.shape).astype(np.float64)
  img_dispR = np.zeros(imgR.shape).astype(np.float64)
  imgL_I = cv2.copyMakeBorder(imgL, win//2, win//2, win//2, win//2, cv2.BORDER_REFLECT)
  imgR_I = cv2.copyMakeBorder(imgR, win//2, win//2, win//2, win//2, cv2.BORDER_REFLECT)

  C = np.zeros((imgR.shape[1], imgL.shape[1]))
  M = np.ones(C.shape)
  C[0][0] = -1
  #since both left and right are of same dimensions I have used shape of Left image in all the code
  for i in range(imgL.shape[0]):
    for row in range(imgL.shape[1]):
      C[row][0] = (row+1)*occlusion_cost
    for col in range(imgL.shape[1]):
      C[0][col] = (col+1)*occlusion_cost
    for row in range(imgL.shape[1]-1):
      for col in range(imgL.shape[1]-1):#same size imgL and imgR
            temp =  NCC(imgL_I[i:i+win, col:col+win],imgR_I[i:i+win, row:row+win])
            min1 = C[row][col]-temp
            min2 = C[row][col+1]+occlusion_cost
            min3 = C[row+1][col]+occlusion_cost
            cmin = min([min1,min2,min3])
            C[row+1][col+1] = cmin; # Cost Matrix
            if cmin==min1:
                M[row+1][col+1] = 1 #Path Tracker
            elif cmin==min2:
                M[row+1][col+1] = 2
            elif cmin==min3:
                M[row+1][col+1] = 3
    r = imgL.shape[1]-1
    c = imgL.shape[1]-1 #since both left and right are of same dimensions
    while r!=0 and c!=0:
      if M[r][c]==1:
        img_dispL[i][c] = abs(r-c)# Disparity part in Left Image
        img_dispR[i][r] = abs(c-r)# Disparity part in Right Image
        r = r-1
        c = c-1
      elif M[r][c]==2:
        img_dispL[i][c] = max_disp #occluded points is assumed to be far away. Thus putting high disparity
        img_dispR[i][r] = max_disp
        r = r-1
      elif M[r][c]==3:
        img_dispR[i][r] = max_disp
        img_dispL[i][c] = max_disp
        c = c-1
  



  ##########-------END OF CODE-------##########
  return img_dispL, img_dispR 