from InfEngine import InfEngine
import numpy as np
import matplotlib.pyplot as plt

def gen_float(exponent, mantissa, scaling = 1):
  ###
  # This function simulates the IEEE754 floating point convention of (-1)^sign * 2^exp * (1.b0b1b2)
  ###
  values = np.array([])
  minW = -pow(2,exponent-1)+1
  maxW = pow(2,exponent-1)+1
  if exponent==0:
    minW = 0
    maxW = 1
  for w in range(minW, maxW):
    for m in range(pow(2,mantissa)):
      acc = 0
      b = "{0:b}".format(m)
      for cnd in range(len(b)):
        if(b[cnd]=='1'):
          acc+=pow(2,-(len(b)-cnd))
        values = np.insert(values,0, pow(2,w)*(1+acc))
  values = np.insert(values, 0, -values)
  values = np.sort(values)	
  values = values*scaling
  return values

def gen_fixed(numb_bits, scaling=1):
  ###
  # This function simulates the normal fixed-point between [0,1), with 0.b0b1b2b3b4...
  ###
  values = np.array([])
  maxV = pow(2,numb_bits)
  for w in range(maxV):
    values = np.insert(values, 0, w/maxV)
  values = np.insert(values,0,-values)
  values = values*scaling
  return values

def gen_fixfl(exponent, mantissa, scaling=1):
  ###
  # This function uses floating point for all values apart from the lowest exponent, which uses fixed point 
  ###
  values = np.array([])
  minW = -pow(2, exponent-1)+1
  maxW = pow(2, exponent-1)+1
  if exponent==0:
    minW = 0
    maxW = 1
  for w in range(minW, maxW):
    # Here we add the condition that if the exponent is at the minimum, then fixed point should be used
    if(w == minW):
      v = gen_fixed(mantissa)
      v = v*pow(2, minW+1)
      values = np.insert(values, 0, v)
      continue	
    for m in range(pow(2, mantissa)):
      acc = 0
      b = "{0:b}".format(m)
      for cnd in range(len(b)):
        if(b[cnd]=='1'):
          acc+=pow(2, -(len(b)-cnd))
      values = np.insert(values, 0, pow(2,w)*(1+acc))
      values = np.insert(values, 0, -pow(2,w)*(1+acc))
  values = np.sort(values)
  values = values*(scaling/np.amax(values))
  return values	


def plot(mantissa_width, exponent_width, nmb_bits, values):	
  fp = gen_float(exponent_width,mantissa_width)
  fxd = gen_fixed(nmb_bits)
  fpxd = gen_fixfl(exponent_width, mantissa_width)
  for_Plot = np.array([values[i] for i in range(len(values)) if i%1000==0])
  fig = plt.figure(figsize=(10,6))
  plt.hist(for_Plot, bins='auto', range=(-1,1), alpha=0.2)
  for xc in fp:
    line1 = plt.axvline(x=xc, color='r', alpha=1, linestyle='-', lw=1, ymax=0.5)
  for xd in fxd:
    line2 =plt.axvline(x=xd, color='b', alpha=1, linestyle='--', lw=1, ymin = 0.25, ymax = 0.75)
  for xf in fpxd:
    line3 = plt.axvline(x=xf, color='k', alpha=1, linestyle='-', lw=1, ymin = 0.5)
  lgd = plt.legend((line1, line2, line3), ('Floating-Point (exp ' + str(exponent_width)+ ', mt ' + str(mantissa_width)+ ')', 'Fixed-Point ' + str(nmb_bits) + '-bit', 'Fixed-Floating Point'))
  lgd.get_frame().set_alpha(1)
  plt.yscale('log')
  plt.xlim((-1,1))
  plt.title("Distribution of values for Floating Point and Fixed Point")
  plt.xlabel("Values")
  return fig

#infer = InfEngine("/home/marcelo/tensorflow/Scripts/MUtils/config/model_setup.yaml")
#values = infer.print_weights()
'''
ts = gen_fixfl(0,3,1)
print("This is the fixfl: ")
print(ts)
print(len(ts))
print("This is the float: ")
ts = gen_float(0,3,1)
print(ts)
print(len(ts))
'''
#plot(2,2,3,4,values)
#plt.show()


