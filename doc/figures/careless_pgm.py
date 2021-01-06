import daft
from matplotlib import pyplot as plt
from IPython import embed


cbrewer1 = '#1b9e77'
cbrewer2 = '#d95f02'
cbrewer3 = '#7570b3'

f_color = 'k'
sig_color = 'k'


p = daft.PGM()


p.add_node('M', '$M$', x=-2, y=0, observed=True)
p.add_node('NN', '$NN_w$', x=-1, y=0, fixed=True, scale=6., label_params={'color': 'w'}, offset=(0., -8.))
p.add_node('mu', '$\mu$', x=0, y=0.5, fixed=True, scale=4., label_params={'color': 'w'}, offset=(0., -8.))
p.add_node('sigma', '$\sigma$', x=0, y=-0.5, fixed=True, scale=4., label_params={'color': 'w'}, offset=(0., -7.))
p.add_edge('NN', "mu")
p.add_edge('NN', "sigma")
p.add_edge('M', "NN")


p.add_node("Sigma", r"$\Sigma$", x = 1, y = 0, plot_params={'ec':sig_color}, label_params={'c' : sig_color})
p.add_edge(f"mu", "Sigma")
p.add_edge(f"sigma", "Sigma")

p.add_node(f"I", "$I_{h,i}$", x =2, y = 0, observed=True)
p.add_edge("Sigma", "I", plot_params={'ec' : sig_color, 'color': sig_color, 'facecolor': sig_color})

p.add_node(f"F", "$F_{h}$", x = 3, y = 0, plot_params={'ec':f_color}, label_params={'c' : f_color})
p.add_edge("F", "I", plot_params={'ec':f_color,'color':f_color,'facecolor':f_color})

p.add_plate([1.35, -1.5 + 0.5, 2.0, 1.6], 'Miller index $h$')
p.add_plate([1.5, -1.5 + 0.90, 1, 1], 'image $i$')

p.render()
plt.savefig('pgm.png', dpi=300)
plt.savefig('pgm.svg', fmt='svg')
plt.show()



p = daft.PGM()


p.add_node("Sigma", r"$\Sigma$", x = 1, y = 0, plot_params={'ec':sig_color}, label_params={'c' : sig_color})

p.add_node(f"I", "$I_{h,i}$", x =2, y = 0, observed=True)
p.add_edge("Sigma", "I", plot_params={'ec' : sig_color, 'color': sig_color, 'facecolor': sig_color})

p.add_node(f"F", "$F_{h}$", x = 3, y = 0, plot_params={'ec':f_color}, label_params={'c' : f_color})
p.add_edge("F", "I", plot_params={'ec':f_color,'color':f_color,'facecolor':f_color})

p.add_plate([1.35, -1.5 + 0.5, 2.0, 1.6], 'Miller index $h$')
p.add_plate([1.5, -1.5 + 0.90, 1, 1], 'image $i$')

p.render()
plt.savefig('simple_pgm.png', dpi=300)
plt.savefig('simple_pgm.svg', fmt='svg')
plt.show()



p = daft.PGM()


p.add_node("Sigma", r"$\Sigma$", x = 1, y = 0, plot_params={'ec':sig_color}, label_params={'c' : sig_color})

p.add_node(f"I", "$I_{h,i}$", x =2, y = 0, observed=True)
p.add_edge("Sigma", "I", plot_params={'ec' : sig_color, 'color': sig_color, 'facecolor': sig_color})

p.add_node(f"F", "$F_{h}$", x = 3, y = 0, plot_params={'ec':f_color}, label_params={'c' : f_color})
p.add_edge("F", "I", plot_params={'ec':f_color,'color':f_color,'facecolor':f_color})

p.add_node("a", r"$a_i$", x = 2, y = 1.0, fixed=True)
p.add_edge("a", "I")

p.add_plate([1.35, -1.5 + 0.5, 2.0, 1.6], 'Miller index $h$')
p.add_plate([1.5, -1.5 + 0.90, 1, 2.0], 'image $i$')

p.render()
plt.savefig('local_pgm.png')
plt.savefig('local_pgm.svg')
plt.show()


