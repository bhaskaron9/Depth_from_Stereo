import numpy as np
def get_depth_map(img_dispL, baseline, focalL):
  # get depth map from disparity
  # be careful with dividing by 0
  img_depthL=np.zeros(img_dispL.shape).astype(np.float64)
  img_depthL[img_dispL>0]=(focalL*baseline)/(img_dispL[img_dispL>0])
  img_depthL[img_dispL==0] = np.max(img_depthL) #for divided by zero

  return img_depthL

def NCC(src, dst):
  # remember to translate so that mean is zero
  src = src.astype(np.float32)
  dst = dst.astype(float)
  src=  src - np.mean(src)#translate
  dst = dst - np.mean(dst)

  ncc = (np.sum(src * dst))/((np.linalg.norm(src)*(np.linalg.norm(dst)))+1e-12)
  return ncc
