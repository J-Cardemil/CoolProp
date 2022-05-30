import CoolProp
import matplotlib.pyplot as plt
from CoolProp.Plots import PropertyPlot
from CoolProp.Plots import SimpleRankineCycle


pp = PropertyPlot('HEOS::Water', 'TS', unit_system='EUR')
pp.calc_isolines(CoolProp.iQ, num=11)

cycle = SimpleRankineCycle('HEOS::Water', 'TS', unit_system='EUR')

T0 = 300
pp.state.update(CoolProp.QT_INPUTS,0.0,T0+15)
p0 = pp.state.keyed_output(CoolProp.iP)

T2 = 700
pp.state.update(CoolProp.QT_INPUTS,1.0,T2-150)
p2 = pp.state.keyed_output(CoolProp.iP)

cycle.simple_solve(T0, p0, T2, p2, 0.7, 0.8, SI=True)
cycle.steps = 50

sc = cycle.get_state_changes()

plt.close(cycle.figure)
pp.draw_process(sc)