import numpy as np
import matplotlib.pyplot as plt

def model(v,v1,tau1,c,sigma):
    v2 = v1 + 1563.2173499656212
    F = 1 - tau1 * np.exp(-(v-v1) ** 2 / ( 2 * sigma**2)) - c * tau1 * np.exp(-(v - v2)**2 / (2 * sigma**2) )
    return F

v1 = 0.0
tau1 = 0.7
c = 1.1
sigma = 150
v = np.linspace(-2000.,10000.,1000)
F = model(v,v1,tau1,c,sigma)
print(F)
plt.plot(v,F)
plt.xlabel('v')
plt.ylabel('Flux')
plt.ylim(0.0,2.0)
plt.show()
