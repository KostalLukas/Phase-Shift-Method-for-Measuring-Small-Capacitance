import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
import sympy as sym

#%%

#Input the variables and their uncertaitnies in this section

#peak to peak voltage Vgpp in V with its uncerttainty in mV
valVgpp = 2.0496
sigmamVgpp = 5.4145

#peak to peak voltage Vcpp in V with its uncerttainty in mV
valVcpp = 1.3174
sigmamVcpp = 4.0474

#phase difference alpha and its uncertainty in degrees
valAlphaDeg = 129.68
sigmaAlphaDeg = 0.60

#resistance of resistor R_1 and its uncertainty in Ohms
valR_1 = 6.8e3
errR_1 = 6.8e1

#frequency f of the driving signal in Hz
valf = 5e4
errf = 0 #assume error in driving frequency is negligable

#capacitance of oscilloscope and its uncertainty in F
valC_2 = 9e-12
errC_2 = 2e-12

#%%

#convert uncertainties in peak to peak voltages from mV to V
sigmaVgpp = sigmamVgpp * 1e-3
sigmaVcpp = sigmamVcpp * 1e-3

#convert alpha and the uncertainty from degrees to radians
valAlphaRad = (valAlphaDeg/180)* sp.pi
sigmaAlphaRad = (sigmaAlphaDeg/180)* sp.pi

#define variables of the function used to find C_1
Vgpp, Vcpp, R_1 ,alphaRad, C_2, f = sym.symbols('Vgpp Vcpp R_1 alphaRad C_2 f', real=True)

#define function to find the value for C_1
C_1func = Vgpp/Vcpp * sym.sin(alphaRad)/(2*sp.pi*f*R_1) - C_2

#find partial derivatives for error propagation
C_1wrtVgpp = sym.diff(C_1func, Vgpp)
C_1wrtVcpp = sym.diff(C_1func, Vcpp)
C_1wrtR_1 = sym.diff(C_1func, R_1)
C_1wrtAlphaRad = sym.diff(C_1func, alphaRad)
C_1wrtC_2 = sym.diff(C_1func, C_2)
C_1wrtf = sym.diff(C_1func, f)

#evaluate the function to find the value of C_1 in F
C_1 = C_1func.subs({Vgpp:valVgpp, Vcpp:valVcpp, R_1:valR_1, alphaRad:valAlphaRad, C_2:valC_2, f:valf})

#evaluate the partial derivatives for error propagation
valC_1wrtVgpp = C_1wrtVgpp.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})
valC_1wrtVcpp = C_1wrtVcpp.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})
valC_1wrtR_1 = C_1wrtR_1.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})
valC_1wrtAlphaRad = C_1wrtAlphaRad.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})
valC_1wrtC_2 = C_1wrtC_2.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})
valC_1wrtf = C_1wrtf.subs({Vgpp:valVgpp, Vcpp:valVcpp,R_1:valR_1, alphaRad:valAlphaRad,C_2:valC_2 ,f:valf})

#array with absolute uncertainties for error propagation
sigmas = np.array([sigmaVgpp, sigmaVcpp, errR_1, sigmaAlphaRad, errC_2, errf])
#array with values of partial derivatives for error propagation
partials = np.array([valC_1wrtVgpp, valC_1wrtVcpp, valC_1wrtR_1, valC_1wrtAlphaRad, valC_1wrtC_2, valC_1wrtf])

#add the uncertainties in quadrature to obtain estimated uncertainty in C_1 in F
errC_1 = np.sqrt(float(np.sum((sigmas*partials)**2)))

#convert the value of C_1 and its estimated uncertainty to pF
CpF = C_1 * 1e12
errCpF = errC_1 * 1e12

#%%

n=20

#define function for calcualting C_1 to use for the plot
def C_1(V_g, V_c, R_1, alphaRad, C_2, f):
    C_1=V_g/(2 * sp.pi * f* V_c *R_1) * np.sin(alphaRad) -C_2
    return C_1

#create array with integers from 0 to n for plotting
N = np.linspace(0, n, n)
#create array to plot dotted line representing the absolute value of C_1 in pF
varCpF = CpF * np.ones(n)

#create arrays with variables varyinglinearly in the range of their uncertainties
varVgpp = np.linspace(valVgpp - sigmaVgpp, valVgpp + sigmaVgpp, n)
varVcpp = np.linspace(valVcpp - sigmaVcpp, valVcpp + sigmaVcpp, n)
varR_1 = np.linspace(valR_1 - errR_1, valR_1 + errR_1, n)
varAlphaRad = np.linspace(valAlphaRad - sigmaAlphaRad, valAlphaRad + sigmaAlphaRad, n)
varC_2 = np.linspace(valC_2 - errC_2, valC_2 + errC_2, n)

#evaulate the value of C_1 for each elent in the array of varied variables in pF
C_1varVgpp = C_1(varVgpp, valVcpp, valR_1, valAlphaRad, valC_2, valf) *1e12
C_1varVcpp = C_1(valVgpp, varVcpp, valR_1, valAlphaRad, valC_2, valf) *1e12
C_1varR_1 = C_1(valVgpp, valVcpp, varR_1, valAlphaRad, valC_2, valf) *1e12
C_1varAlpha = C_1(valVgpp, valVcpp, valR_1, varAlphaRad, valC_2, valf) *1e12
C_1varC_2 = C_1(valVgpp, valVcpp, valR_1, valAlphaRad, varC_2, valf) *1e12

#set fonts for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = "cm"

#create plot title and label axis
plt.title('Variantion of $C_2$')
plt.ylabel('$C_2$/pF')
plt.xlabel('variation of variables over the range of their uncertainties')

#plot the valkue of C_1 against each variable in the range of its uncertainties
plt.plot(N, C_1varVgpp, color='blue', label=r'$V_g$')
plt.plot(N, C_1varVcpp, color='red', label=r'$V_c$')
plt.plot(N, C_1varR_1, color='cyan', label=r'$R_1$')
plt.plot(N, C_1varAlpha, color='magenta', label=r'$ \alpha $')
plt.plot(N, C_1varC_2, color='lime', label=r'$C_2$')
plt.plot(N, varCpF, '--',color='black')

#make a plot legend and save the plot
plt.legend()
plt.savefig('Plots/Variable Dependance.png', dpi=200)
plt.show()

#print the final calulated value and uncertainty for C_1 in pF
print('----------------------------------------------')
print('Capacitance C_1 = %.2f ± %.2f pF' % (CpF, errCpF))
print('----------------------------------------------')