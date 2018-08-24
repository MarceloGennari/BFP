import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Setting variables")
parser.add_argument("-m", "--mantissa", help="Specifies the width of the mantissa", type=int)
parser.add_argument("-x", "--exponent", help="Specifies the width of the exponent", type=int)
args = parser.parse_args()

m_w = args.mantissa
e_w = args.exponent

# reads tmp file
f = open('tmp.txt', 'r')
next(f)
f = f.read()
# gets all the numbers on that
nmb = re.findall(r"[-+]?\d*\.\d+|\d+", f)
nmb = [float(i) for i in nmb if i!='0']
nmb = np.array(nmb)
nmb = np.sort(nmb)
maxv = np.amax(nmb)
minv = np.amin(nmb)

plt.hist(nmb, bins='auto')
plt.title("Distribution for Mant=%i Exp=%i" % (m_w, e_w))
plt.xlabel("Tensor Values")
plt.annotate('max: %f' %maxv, xy=(0.7, 0.8), xycoords='axes fraction')
plt.annotate('min: %f' %minv, xy=(0.7, 0.7), xycoords='axes fraction')
plt.savefig("/mnt/d/TensorResults/m%ix%i.png" %(m_w, e_w))
