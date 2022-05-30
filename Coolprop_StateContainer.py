from __future__ import print_function
import CoolProp
import CoolProp.CoolProp as cp
from CoolProp.Plots.SimpleCycles import StateContainer


T0 = 300
p0 = 200000
d0 = cp.PropsSI('D','T',T0,'P',p0,'Water')
h0 = 112745.749
s0 = 393.035

cycle_states = StateContainer()

cycle_states[0,'H'] = h0
cycle_states[0]['S'] = s0
cycle_states[0]['D'] = d0
cycle_states[0][CoolProp.iP] = p0
cycle_states[0,CoolProp.iT] = T0
cycle_states[1,"T"] = 300.064

print(cycle_states)