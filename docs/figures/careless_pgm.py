import daft
from matplotlib import pyplot as plt
from IPython import embed


f_color = 'k'
sig_color = 'k'


p = daft.PGM()


p.add_node('M', '$M$', x=-0.33, y=0, observed=True)


p.add_node("Sigma", r"$\Sigma_{h,i}$", x = 1, y = 0, plot_params={'ec':sig_color}, label_params={'c' : sig_color})
p.add_edge('M', "Sigma", label=r'$\theta$', xoffset=-0.33)

p.add_node(f"I", "$I_{h,i}$", x =2, y = 0, observed=True)
p.add_edge("Sigma", "I", plot_params={'ec' : sig_color, 'color': sig_color, 'facecolor': sig_color})

p.add_node(f"F", "$F_{h}$", x = 3, y = 0, plot_params={'ec':f_color}, label_params={'c' : f_color})
p.add_edge("F", "I", plot_params={'ec':f_color,'color':f_color,'facecolor':f_color})

p.add_plate([0.33, -1.0, 3.0, 1.6], 'Miller index $h$', position="bottom right")
p.add_plate([0.5, -0.6, 2.0, 1], 'image $i$', position="bottom right")

p.render()
plt.savefig('panels/pgm.png', dpi=300)
plt.savefig('panels/pgm.svg', fmt='svg')
plt.show()


p = daft.PGM()


#p.add_node('M', '$M$', x=-0.33, y=0, observed=True)


p.add_node("Sigma", r"$\Sigma_{h,i}$", x = 1, y = 0, plot_params={'ec':sig_color}, label_params={'c' : sig_color})
#p.add_edge('M', "Sigma", label=r'$\theta$', xoffset=-0.33)

p.add_node(f"I", "$I_{h,i}$", x =2, y = 0, observed=True)
p.add_edge("Sigma", "I", plot_params={'ec' : sig_color, 'color': sig_color, 'facecolor': sig_color})

p.add_node(f"F", "$F_{h}$", x = 3, y = 0, plot_params={'ec':f_color}, label_params={'c' : f_color})
p.add_edge("F", "I", plot_params={'ec':f_color,'color':f_color,'facecolor':f_color})

p.add_plate([0.33, -1.0, 3.0, 1.6], 'Miller index $h$', position="bottom right")
p.add_plate([0.5, -0.6, 2.0, 1], 'image $i$', position="bottom right")

p.render()
plt.savefig('panels/pgm_simple.png', dpi=300)
plt.savefig('panels/pgm_simple.svg', fmt='svg')
plt.show()

