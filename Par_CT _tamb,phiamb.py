# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:02:30 2022

@author: jblan
"""

import CoolProp.CoolProp as CP
from CoolProp.HumidAirProp import HAPropsSI
import math
from scipy import optimize as op
import numpy as np
import matplotlib.pyplot as plt

# ------------- inputs
P_amb = 101325 #Pa Atmosferic pressure (mantained constant)

# Tower coefficients (From Hyhlík 2017, for a natural draft WCT, 
#                     and in the range reported in Chargui et al 2013)
c = 2.33
n = -1.1
Dia = 25 # Selected fan diameter
theta = 13 #6-15 # pitch angle of the fan
L_fi = 22 #20-28 ft, fill height

# ----------- Condenser of Rankine cycle (From EES model)
P_c = 4500 #Pa Condenser Pressure Rankine side
x_13 = 0.9446 # Quality at the turbine outlet
T_c = CP.PropsSI('T','Q',x_13,'P',P_c,'Water')
h_13 = CP.PropsSI('H','Q',x_13,'P',P_c,'Water') # Condenser inlet steam enthalpy
h_14 = CP.PropsSI('H','Q',0,'P',P_c,'Water') # Condenser outlet steam enthalpy
h_35 = h_36 = h_37 = CP.PropsSI('H','Q',0,'P',11160,'Water') # Condenser intermediate enthalpy
m_dot_c1 = 91.09 #kg/s Condenser inlet steam flow mass
m_dot_c2 = 106.1 #kg/s Condenser outlet steam flow mass
Q_dot_c1 = m_dot_c1*(h_13 - h_36)
Q_dot_c2 = m_dot_c2*(h_37 - h_14)
Q_dot_c = Q_dot_c1 + Q_dot_c2 #J/s Heat rate rejection required at the condenser

# ----------- Integration Condenser-CT

# --- Water side
TTD_c = 5 #C TTD in the condenser
T_w1 = T_c - TTD_c #K Water inlet temperature (assumed constant)
Range_c = 10 #C Initial guess for CT range
T_w2_i = T_w1 - Range_c #K Initial guess for water outlet temperature
Cp_w = CP.PropsSI('C','T',T_w1,'P',P_amb,'Water') #J/kg-K Specific heat of water
m_dot_w_i = Q_dot_c/(Cp_w*(T_w1 - T_w2_i)) #kg/s Initial gues for mass flow rate of water
# --- Air side
h_a2_i = hs_w1 = HAPropsSI('H','T',T_w1,'P',P_amb,'R',1) #J/kg dry air 
#               Enthalpy of saturated air at water inlet temperature, used as
#               initial guess for outlet air condition
T_amb_i = 25 + 273.15 #K Air dry bulb temperature
phi_amb_i = 30/100 # Relative humidity of ambient air

T_a1_i = T_amb_i    # Inlet air temperature
h_a1_i = HAPropsSI('H','T',T_a1_i,'P',P_amb,'R',phi_amb_i) # Inlet air enthalpy
m_dot_a_i = 1.4*Q_dot_c/(h_a2_i - h_a1_i) # Mass flow of air (must be established)

#%%
# ----------- Function for water outlet temperature iteration
# Inputs:
#   - T_aux, Initial guess for water outlet temperature
#   - Ntu, calculated with tower coefficients and air and water mass flows
#   - m_dot_w, mass flow of water
#   - m_dot_a, mass flow of air
#   - T_a1_f, inlet Air temperature
#   - h_a1, inlet air enthalpy
#   - phi_amb_f, inlet air relative humidity
def T_w_outlet(T_aux,Ntu,m_dot_w,m_dot_a,T_a1_f,h_a1_f,phi_amb_f):

    global m_dot_mkw_i # Mass flow of make-up (update does not work sometimes, that is why the calculation is repeated afterwards)
    
    # Calculation of air side heat rate transfer and outlet properties
    hs_w2 = HAPropsSI('H','T',T_aux,'P',P_amb,'R',1) # Enthalpy of saturated air at water outlet temperature
    C_s = (hs_w1 - hs_w2)/(T_w1 - T_aux) #    
    m_star = (m_dot_a*C_s)/(m_dot_w*Cp_w) #
    eff_a = ((1 - math.exp(-Ntu*(1 - m_star))) # Air side heat transfer effectiveness
         /(1 - m_star*math.exp(-Ntu*(1 - m_star))))
    Q_dot_1 = eff_a*m_dot_a*(hs_w1-h_a1_f) # Calculated heat transfer with respect to theoric maximun air enthalpy difference
    h_a2 = Q_dot_1/m_dot_a + h_a1_f # Calculated air outlet enthalpy
    
    # Calculation of evaporated water mass flow for water side heat rate and comparison
    h_eff = h_a1_f + (h_a2 - h_a1_f)/(1 - math.exp(-Ntu)) #
    omega_eff = HAPropsSI('W','H',h_eff,'P',P_amb,'R',1) #
    omega_a1 = HAPropsSI('W','T',T_a1_f,'P',P_amb,'R',phi_amb_f) # Inlet air humidity ratio
    omega_a2 = omega_eff + (omega_a1-omega_eff)*math.exp(-Ntu) # Outlet air humidity ratio
    m_dot_w1 = m_dot_w # Inlet water flow mass
    m_dot_w2 = m_dot_w1 - m_dot_a*(omega_a2-omega_a1) # Outlet water flow mass
    m_dot_mkw_i = m_dot_w1 - m_dot_w2 # Mass flow of make-up water
    Q_dot_w = Cp_w*(m_dot_w1*(T_w1-273.15) - m_dot_w2*(T_aux-273.15)) # Calculated water side heat transfer

    return abs(Q_dot_1 - Q_dot_w) # Function to compare

# ----------- Function for water flow mass iteration
# Inputs:
#   - m_dot_w_guess, Initial guess for water inlet flow mass
#   - m_dot_a_f, Mass flow of air
#   - T_a1_f, inlet air temperature
#   - h_a1_f, inlet air enthalpy
#   - phi_amb_f, inlet air relative humidity

def m_dot_w_it(m_dot_w_guess,m_dot_a_f,T_a1_f,h_a1_f,phi_amb_f):
  
    global T_w2_f
    Ntu_1 = c*(m_dot_w_guess/m_dot_a_f)**(1+n) # Calculation of NTU
    T_w2_guess = 15 + 273.15 #K Initial temperature guess
    T_w2_f = op.newton(T_w_outlet,T_w2_guess,args=(Ntu_1,m_dot_w_guess,m_dot_a_f,T_a1_f,h_a1_f,phi_amb_f)) #K Call temperature iterative function
    
    # Calculation is repeated considering problems with update of global variables
    hs_w2 = HAPropsSI('H','T',T_w2_f,'P',P_amb,'R',1) # Enthalpy of saturated air at water outlet temperature
    C_s = (hs_w1 - hs_w2)/(T_w1 - T_w2_f) #    
    m_star = (m_dot_a_f*C_s)/(m_dot_w_guess*Cp_w) #
    eff_a = ((1 - math.exp(-Ntu_1*(1 - m_star))) # Air side heat transfer effectiveness
          /(1 - m_star*math.exp(-Ntu_1*(1 - m_star))))
    Q_dot_1 = eff_a*m_dot_a_f*(hs_w1-h_a1_f) # Calculated heat transfer with respect to theoric maximun air enthalpy difference
    h_a2 = Q_dot_1/m_dot_a_f + h_a1_f # Calculated air outlet enthalpy
    
    h_eff = h_a1_f + (h_a2 - h_a1_f)/(1 - math.exp(-Ntu_1)) #
    omega_eff = HAPropsSI('W','H',h_eff,'P',P_amb,'R',1) #
    omega_a1 = HAPropsSI('W','T',T_a1_f,'P',P_amb,'R',phi_amb_f) # Inlet air humidity ratio
    omega_a2 = omega_eff + (omega_a1-omega_eff)*math.exp(-Ntu_1) # Outlet air humidity ratio
    m_dot_w1 = m_dot_w_guess # Inlet water flow mass
    m_dot_w2 = m_dot_w1 - m_dot_a_f*(omega_a2-omega_a1) # Outlet water flow mass

    Q_dot_w = Cp_w*(m_dot_w1*(T_w1-273.15) - m_dot_w2*(T_w2_f-273.15))
    #print(m_dot_w_guess)
    return abs(Q_dot_c - Q_dot_w) # Function to compare

# ------------ Fan and pump power requirements
#
# ---- Fan Power
# OSTI method
def P_fan_1(m_dot_a,h_a2,omega_a2):
    
    G_a = m_dot_a*2.20462*3600 #lbs/hr
    rho_a2 = 1/HAPropsSI('Vha','Hha',h_a2,'P',P_amb,'W',omega_a2)
    rho_mix = rho_a2*2.20462/(3.28084**3) #lbs/ft^3
    F_acfm = (1 + omega_a2)*G_a/(60*rho_mix) #acfm, 
    P_fan_1_hp =  F_acfm/8000 #hp
    P_fan_1 = P_fan_1_hp*0.0007457 #MW
    
    return P_fan_1

# Kloppers method
def P_fan_2(m_dot_a,h_a2,omega_a2):
    
    m_dot_av2 = m_dot_a*(1 + omega_a2)
    rho_a2 = 1/HAPropsSI('Vha','Hha',h_a2,'P',P_amb,'W',omega_a2)
    V_flow_a = m_dot_av2/rho_a2 #m^3/s
    Vel = V_flow_a/(np.pi*(1/4)*Dia**2) #m/s
    rpm = Vel*60/(2*np.pi*(Dia/2-1.5)*np.sin(np.deg2rad(theta)))
    rpm_ref = 750
    dia_ref = 1.536
    rho_a_ref = 1.2
    V_flow_dif = V_flow_a*(rpm_ref/rpm)*(dia_ref/Dia)**3
    P_fan_dif = 4245.1 - 64.134*V_flow_dif + 17.586*V_flow_dif**2 - 0.71079*V_flow_dif**3
    P_fan_2 = 1e-6*P_fan_dif*(rho_a2/rho_a_ref)*((rpm/rpm_ref)**3)*(Dia/dia_ref)**5 #MW
    
    return P_fan_2

# ---- Condenser Pump Power
# OSTI method
def P_pump(m_dot_w):
    
    H_p = L_fi + 10

    eta_p_ref = .8
    m_dot_w_ref = m_dot_w_i
    eta_pump = eta_p_ref*(2*(m_dot_w/m_dot_w_ref)-(m_dot_w/m_dot_w_ref)**2) # Patnode eq 3.53
    P_pump_1 = 1e-6*m_dot_w*9.81*(H_p*0.3048)/eta_pump #MW
    
    return P_pump_1

#%%
# ----------- Call water flow mass iterative function

m_dot_w_i = op.newton(m_dot_w_it,m_dot_w_i*1.2,args=(m_dot_a_i,T_a1_i,h_a1_i,phi_amb_i)) #F=1.2

#%%
print("Resultados iniciales de integración del condensador de ciclo Rankine:")
print("Flujo másico de agua calculado:{r:1.2f} kg/s".format(r=m_dot_w_i))
print("Temperatura del agua de salida calculada:{r:1.4f} C".format(r=T_w2_f-273.15))
print("Temperatura del agua de entrada:{r:1.4f} C".format(r=T_w1-273.15))
print("Flujo másico de aire:{r:1.2f} kg/s".format(r=m_dot_a_i))
print("Flujo másico de agua make-up:{r:1.2f} kg/s".format(r=m_dot_mkw_i))


#%%
# -------- Parametrization of air inlet properties

# Mass flow of air and water are already known 

T_amb_t = np.linspace(10+273.15,40+273.15,10) # Ambient temperature
phi_amb_t = np.linspace(15/100,80/100,10) # Relative humidity of air

T_w2_p  = np.zeros(10) # Outlet water temperature
m_dot_mkw_p = np.zeros(10) # Mass flow of make-up water
omega_a1_p = np.zeros(10) # 
omega_a2_p = np.zeros(10)
h_a1_p = np.zeros(10)
h_a2_p = np.zeros(10)
P_pump_p = np.zeros(10)
P_fan_1_p = np.zeros(10)
P_fan_2_p = np.zeros(10)
i = 0

# Two cases:
# 1) T_amb variable and phi_amb constant
# 2) T_amb constant and phi_amb variable

for par_amb in T_amb_t: # Change for T_amb_t or phi_amb_t depending on the case 

#   These uncommented for first case
    T_a1_p = par_amb
    phi_amb_p = 30/100

#   These uncommented for second case    
    # T_a1_p = 25 + 273.15
    # phi_amb_p = par_amb
    
    h_a1_p[i]= HAPropsSI('H','T',T_a1_p,'P',P_amb,'R',phi_amb_p) # inlet air enthalpy
    
    Ntu_p = c*(m_dot_w_i/m_dot_a_i)**(1+n)
    T_w2_guess = 15 + 273.15 #K Initial temperature guess
    T_w2_p[i] = op.newton(T_w_outlet,T_w2_guess,args=(Ntu_p,m_dot_w_i,m_dot_a_i,T_a1_p,h_a1_p[i],phi_amb_p))
    
    # {Calculation afterwards could be avoided when problem with update of global variables is solved
    hs_w2 = HAPropsSI('H','T',T_w2_p[i],'P',P_amb,'R',1) # Enthalpy of saturated air at water outlet temperature
    C_s = (hs_w1 - hs_w2)/(T_w1 - T_w2_p[i]) #    
    m_star = (m_dot_a_i*C_s)/(m_dot_w_i*Cp_w) #
    eff_a = ((1 - math.exp(-Ntu_p*(1 - m_star))) # Air side heat transfer effectiveness
          /(1 - m_star*math.exp(-Ntu_p*(1 - m_star))))
    Q_dot_1 = eff_a*m_dot_a_i*(hs_w1-h_a1_p[i]) # Calculated heat transfer with respect to theoric maximun air enthalpy difference
    h_a2_p[i] = Q_dot_1/m_dot_a_i + h_a1_p[i] # Calculated air outlet enthalpy
    
    h_eff = h_a1_p[i] + (h_a2_p[i] - h_a1_p[i])/(1 - math.exp(-Ntu_p)) #
    omega_eff = HAPropsSI('W','H',h_eff,'P',P_amb,'R',1) #
    omega_a1_p[i] = HAPropsSI('W','T',T_a1_p,'P',P_amb,'R',phi_amb_p) # Inlet air humidity ratio
    omega_a2_p[i] = omega_eff + (omega_a1_p[i]-omega_eff)*math.exp(-Ntu_p) # Outlet air humidity ratio
    m_dot_w1 = m_dot_w_i # Inlet water flow mass
    m_dot_w2 = m_dot_w1 - m_dot_a_i*(omega_a2_p[i]-omega_a1_p[i]) # Outlet water flow mass 
    m_dot_mkw_p[i] = m_dot_w1 - m_dot_w2
    # }
    
#   Calculation of pump/fan power consumption 
    P_pump_p[i] = P_pump(m_dot_w_i)
    P_fan_1_p[i] = P_fan_1(m_dot_a_i,h_a2_p[i],omega_a2_p[i])
    P_fan_2_p[i] = P_fan_2(m_dot_a_i,h_a2_p[i],omega_a2_p[i])
    
    i += 1

#%% ---------------- Parametric T_amb with phi_amb constant
fig, ax1 = plt.subplots(dpi=300) 
ax2 = ax1.twinx()
ax1.plot(T_amb_t-273.15,T_w2_p-273.15,c='b')
ax2.plot(T_amb_t-273.15,m_dot_mkw_p,c='g')
ax1.tick_params(axis='y', color='b', labelcolor='b')
ax2.tick_params(axis='y', color='g', labelcolor='g')
ax1.set_xlabel('Ambient temperature [C]')
ax1.set_ylabel('Water oulet temperature [C]', c='b')
ax2.set_ylabel('Make-up water [kg/s]', c='g')   
ax1.grid()
plt.title(r'$\phi$ = '+f'{phi_amb_p*100:.0f}'+'%')

#%%
fig, ax1 = plt.subplots(dpi=300) 
ax1.plot(T_amb_t-273.15,omega_a1_p,c='b',label='a_i_w')
ax1.plot(T_amb_t-273.15,omega_a2_p,c='g',label='a_o_w')
ax1.plot(T_amb_t-273.15,omega_a2_p-omega_a1_p,c='r',label='dif_w')
ax1.set_xlabel('Ambient temperature [C]')
ax1.set_ylabel('Air Humidity ratio [kg/kg]')
ax1.grid()
ax1.legend()
plt.title(r'$\phi$ = '+f'{phi_amb_p*100:.0f}'+'%')

#%%
fig, ax1 = plt.subplots(dpi=300) 
ax1.plot(T_amb_t-273.15,h_a1_p/1000,c='b',label='a_i_h')
ax1.plot(T_amb_t-273.15,h_a2_p/1000,c='g',label='a_o_h')
ax1.set_xlabel('Ambient temperature [C]')
ax1.set_ylabel('Air Enthalpy [kg/kg]')
ax1.grid()
ax1.legend()
plt.title(r'$\phi$ = '+f'{phi_amb_p*100:.0f}'+'%')

#%%
fig, ax1 = plt.subplots(dpi=300) 
ax1.plot(T_amb_t-273.15,P_pump_p,c='b',label=r'$P_{pump}$')
ax1.plot(T_amb_t-273.15,P_fan_1_p,c='r',label=r'$P_{fan,1}$')
ax1.plot(T_amb_t-273.15,P_fan_2_p,c='g',label=r'$P_{fan,2}$')
ax1.set_xlabel('Ambient temperature [C]')
ax1.set_ylabel('Power consumption [MW]')
ax1.legend()
ax1.grid()
plt.title(r'$\phi$ = '+f'{phi_amb_p*100:.0f}'+r' %') 

# #%% 
# #  Uncomment for second case
# #
# #---------------- Parametric phi_amb with T_amb constant
# fig, ax1 = plt.subplots(dpi=300) 
# ax2 = ax1.twinx()
# ax1.plot(phi_amb_t*100,T_w2_p-273.15,c='b')
# ax2.plot(phi_amb_t*100,m_dot_mkw_p,c='g')
# ax1.tick_params(axis='y', color='b', labelcolor='b')
# ax2.tick_params(axis='y', color='g', labelcolor='g')
# ax1.set_xlabel('Relative humidity [%]')
# ax1.set_ylabel('Water oulet temperature [C]', c='b')
# ax2.set_ylabel('Make-up water [kg/s]', c='g')   
# ax1.grid()
# plt.title(r'$T_{amb}$ = '+f'{T_a1_p-273.15:.0f}'+'[C]')
# #%%
# fig, ax1 = plt.subplots(dpi=300) 
# ax1.plot(phi_amb_t*100,omega_a1_p,c='b',label='a_i_w')
# ax1.plot(phi_amb_t*100,omega_a2_p,c='g',label='a_o_w')
# ax1.plot(phi_amb_t*100,omega_a2_p-omega_a1_p,c='r',label='dif_w')

# ax1.set_xlabel('Relative humidity [%]')
# ax1.set_ylabel('Air Humidity ratio [kg/kg]')
# ax1.grid()
# ax1.legend()
# plt.title(r'$T_{amb}$ = '+f'{T_a1_p-273.15:.0f}'+'[C]')

# #%%
# fig, ax1 = plt.subplots(dpi=300) 
# ax1.plot(phi_amb_t*100,h_a1_p/1000,c='b',label='a_i_h')
# ax1.plot(phi_amb_t*100,h_a2_p/1000,c='g',label='a_o_h')

# ax1.set_xlabel('Relative humidity [%]')
# ax1.set_ylabel('Air Enthalpy [kg/kg]')
# ax1.grid()
# ax1.legend()
# plt.title(r'$T_{amb}$ = '+f'{T_a1_p-273.15:.0f}'+'[C]')

# #%%
# fig, ax1 = plt.subplots(dpi=300) 
# ax1.plot(phi_amb_t*100,P_pump_p,c='b',label=r'$P_{pump}$')
# ax1.plot(phi_amb_t*100,P_fan_1_p,c='r',label=r'$P_{fan,1}$')
# ax1.plot(phi_amb_t*100,P_fan_2_p,c='g',label=r'$P_{fan,2}$')
# # ax1.set_xlim(5800,7400)
# # ax1.set_ylim(0.65,1.3)
# # ax1.set_yticks(np.arange(0.7, 1.3, 0.1))
# ax1.set_xlabel('Relative humidity [%]')
# ax1.set_ylabel('Power consumption [MW]')
# ax1.legend()
# ax1.grid()
# plt.title(r'$T_{amb}$ = '+f'{T_a1_p-273.15:.0f}'+'[C]')