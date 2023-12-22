# Double Wilson Prior Calculations in `careless`

This page briefly describes the so-called "Double Wilson" distributions available for use in `careless` when using one dataset as a reference for another.

We'd like to be able to make priors in which certain sets of structure factors are conditional on others. The conditional distributions for structure factors are the Rice and Woolfson (Folded Normal) distributions for acentrics and centrics respectively. 


## Wilson's priors
\begin{equation*}
\mathcal{W}ilson(F_h | \epsilon_h, \Sigma) = \begin{cases} 
    \mathcal{H}alfnormal\left(F_h|\epsilon_h \Sigma \right) = \sqrt{\frac{2}{\pi\epsilon_h\Sigma}} \exp\left(-\frac{F^2_h}{2\epsilon_h\Sigma}\right)  & h \in centrics \\
    \mathcal{R}ayleigh\left(F_h| \epsilon_h \Sigma \right) = \frac{2}{\epsilon_h\Sigma} F_h \exp\left(-\frac{F^2_h}{\epsilon_h\Sigma}\right) & h \in acentrics \\
\end{cases}
\end{equation*}


## Joint priors

For two structure factors, $G_h$ and $F_h$, let

\begin{align*}
    z_f &\sim F_h \\
    z_g &\sim G_h \\
\end{align*}


denote reparameterized samples. 
The joint probability factorizes as, 
$$P(F_h, G_h) = P(G_h | F_h) P(F_h).$$ 
$F_h$ follows Wilson's prior, 
$$P(F_h) = \mathcal{W}ilson(\epsilon_h, \Sigma),$$


## Acentric joint priors
The joint probability of the samples is

\begin{align*}
P(z_f, z_g| r, \Sigma, \epsilon_h) &= \mathcal{R}ice\left(z_g \big\vert r z_f, \sqrt{\frac{\epsilon_h \Sigma }{2}\left(1 - r^2\right)}\right) 
\mathcal{W}ilson\left(z_f|\epsilon_h, \Sigma \right),\\
\end{align*}

Where the conditional is a Rice distribution,
$$
Rice(x | \nu, \sigma) = \frac{x}{\sigma^2} \exp\left(\frac{-(x^2+\nu^2)}{2\sigma^2}\right)I_0\left(\frac{x\nu}{\sigma^2}\right)
$$



## Centric joint priors
The joint probability of the samples is

\begin{align*}
P(z_f, z_g| r, \Sigma, \epsilon_h) &= \mathcal{F}oldednormal\left(z_g \big\vert r z_f, \sqrt{\epsilon_h \Sigma \left(1 - r^2\right)}\right) 
\mathcal{W}ilson\left(z_f|\epsilon_h, \Sigma \right),
\end{align*}

Where the conditional is a folded normal distribution,
$$
FoldedNormal(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) + \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x+\mu)^2}{2\sigma^2}\right)
$$
