import daft
from matplotlib import pyplot as plt
from IPython import embed



p = daft.PGM()

d = 4
layers = 5


for i in range(layers):
    for j in range(d):
        p.add_node(f'{i}_{j}', x=i, y=j, fixed=True)
    if i > 0:
        for j in range(d):
            for k in range(d):
                p.add_edge(f'{i}_{j}', f'{i-1}_{k}', directed=False)

for i in range(d):
    p.add_node(f'm_{i}', r'$m_{i}$', x=-1., y=i, observed=True)
    p.add_edge(f"m_{i}", f"{0}_{i}")


p.add_node(f'{layers}_0', x=layers, y=d/2, fixed=True)
p.add_node(f'{layers}_1', x=layers, y=d/2-1.0, fixed=True)

for i in range(d):
    p.add_edge(f"{layers}_0", f"{layers-1}_{i}", directed=False)
    p.add_edge(f"{layers}_1", f"{layers-1}_{i}", directed=False)

p.add_node("Sigma", r"$\Sigma_{h}^{i}$", x = layers + 1, y = (d-1)/2.)
p.add_edge(f"{layers}_0", "Sigma")
p.add_edge(f"{layers}_1", "Sigma")

p.add_node(f"I", "$I_{h}^{i}$", x = layers + 2, y = (d-1)/2., observed=True)
p.add_edge("Sigma", "I")

p.add_node(f"F", "$F_{h}$", x = layers + 3, y = (d-1)/2.)
p.add_edge("F", "I")

p.add_node(f"W", "W", x = layers + 4, y = (d-1)/2.)
p.add_edge("W", "F")

p.add_plate([layers+0.3, 0.2, 3.2, 1.9], 'miller index $h$')
p.add_plate([layers+0.5, 0.80, 2, 1.1], 'image $i$')

p.render()
plt.show()
