def eig_fun_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [(v+p_sqrt)/(2*D)
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return phi

def eig_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [(v+p_sqrt)/(2*D)
         (v-p_sqrt)/(2*D)
    ]

    c = (1 - r[1]/r[0]) * np.exp(r[1]-t*l)

    psi = c * np.exp(t*l*x)

    return psi


def eig_fun_adj_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [-(v+p_sqrt)/(2*D)
         -(v-p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return phi_star

def eig_fun_adj_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [-(v+p_sqrt)/(2*D)
         -(v-p_sqrt)/(2*D)
    ]

    c = 1 - r[1]/r[0]
    
    psi_star = c * np.exp(-t*l*x)

    return psi_star

