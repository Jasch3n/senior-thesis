import numpy as np

def getFourierCoeffs(x, func, num=11):
  num_pts = len(x)
  sin_coeffs = []
  cos_coeffs = []
  for i in range(1,num+1):
    sin_coeffs.append(sum(func(x) * np.sin(i*np.pi*x)) * ((np.max(x)-np.min(x))/num_pts))
    cos_coeffs.append(sum(func(x) * np.cos(i*np.pi*x)) * ((np.max(x)-np.min(x))/num_pts))
  zero_cos_coeff = sum(func(x)) * ((np.max(x)-np.min(x))/num_pts)
  return (sin_coeffs, [zero_cos_coeff] + cos_coeffs)