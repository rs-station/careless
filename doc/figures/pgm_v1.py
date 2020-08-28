import daft
from matplotlib import pyplot as plt
from IPython import embed



p = daft.PGM()

d = 4
layers = 5
nn_scale = 2.
nn_arrows = False

for i in range(layers):
    for j in range(d):
        p.add_node(f'{i}_{j}', x=i, y=j, fixed=True, scale=nn_scale)
    if i > 0:
        for j in range(d):
            for k in range(d):
                p.add_edge(f'{i-1}_{k}', f'{i}_{j}', directed=nn_arrows)

for i in range(d):
    p.add_node(f'm_{i}', f'$m_{i}$', x=-1., y=i, observed=True)
    p.add_edge(f"m_{i}", f"{0}_{i}", directed=nn_arrows)


p.add_node(f'{layers}_0', '$\mu$', x=layers, y=d/2, fixed=True, scale=2.*nn_scale, label_params={'color': 'w'}, offset=(0., -6.))
p.add_node(f'{layers}_1', '$\sigma$', x=layers, y=d/2-1.0, fixed=True, scale=2.*nn_scale, label_params={'color': 'w'}, offset=(0., -7.))

for i in range(d):
    p.add_edge(f"{layers-1}_{i}", f"{layers}_0", directed=nn_arrows)
    p.add_edge(f"{layers-1}_{i}", f"{layers}_1", directed=nn_arrows)

p.add_node("Sigma", r"$\Sigma$", x = layers + 1, y = (d-1)/2.)
p.add_edge(f"{layers}_0", "Sigma")
p.add_edge(f"{layers}_1", "Sigma")

p.add_node(f"I", "$I_{h}^{i}$", x = layers + 2, y = (d-1)/2., observed=True)
p.add_edge("Sigma", "I")

p.add_node(f"F", "$F_{h}$", x = layers + 3, y = (d-1)/2.)
p.add_edge("F", "I")

p.add_node(f"W", "W", x = layers + 4, y = (d-1)/2.)
p.add_edge("W", "F")

p.add_plate([layers+1.35, 0.3, 2.2, 2], 'miller index $h$')
p.add_plate([layers+1.5, 0.90, 1, 1], 'image $i$')

p.render()
plt.show()
