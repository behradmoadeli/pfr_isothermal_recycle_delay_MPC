def eig_fun_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][0]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return b * phi

def eig_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][0]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    c = (1 - r[1]/r[0]) * np.exp(r[1]-t*l)

    psi = c * np.exp(t*l*x)

    return b * psi


def eig_fun_adj_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1].conjugate()

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][1]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return b * phi_star

def eig_fun_adj_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1].conjugate()

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][1]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    c = R * v * t * (1 - r[1]/r[0])
    
    psi_star = c * np.exp(-t*l*x)

    return b * psi_star

def eig_fun_1_prime(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][0]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * r[0] * np.exp(r[0]*x) + r[1] * np.exp(r[1]*x)

    return b * phi

def eig_fun_adj_1_prime(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1].conjugate()

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(complex(p))
    
    try:
        b = args[2][1]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * r[0] * np.exp(r[0]*x) + r[1] * np.exp(r[1]*x)

    return b * phi_star

def init_cond_func_1(x, *args):
    
    import numpy as np

    # y = np.ones_like(x)
    y = 1-np.cos(2*np.pi*x)
    # par = args[0]
    
    # y = 2 * eig_fun_1(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_1(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)
    
    return y

def init_cond_func_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    # y = (1 + (R-1) * x) / R
    y = np.zeros_like(x)
    # y = 1/R + x * (np.cos(2*np.pi * x) - 1/R)
    # y = 2 * eig_fun_2(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_2(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)

    return y

def q_fun_1(x, *args):
    
    import numpy as np

    y = np.ones_like(x)
    # y = np.cos(2*np.pi*x)
    # par = args[0]
    
    # y = 2 * eig_fun_1(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_1(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)
    
    return y

def q_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    y = np.ones_like(x)
    # y = (1 + (R-1) * x) / R
    # y = 1/R + x * (np.cos(2*np.pi * x) - 1/R)
    # y = 2 * eig_fun_2(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_2(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)

    return y

def arbit_fun_1(x, *args):
    
    import numpy as np

    # y = np.ones_like(x)
    y = np.cos(2*np.pi*x)
    # par = args[0]
    
    # y = 2 * eig_fun_1(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_1(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)
    
    return y

def arbit_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    # y = (1 + (R-1) * x) / R
    y = 1/R + x * (np.cos(2*np.pi * x) - 1/R)
    # y = 2 * eig_fun_2(x, par, 0.6409081359945565+0j, -0.47362647607303926j) + 7* eig_fun_2(x, par, -2.470670435901706+8.418531957532583j, 0.47930716047278943-0.0018801497185718524j)

    return y

def eig_fun_mul_0(x, *args):
    
    import numpy as np
    
    par = args[0]
    l = args[1]
    
    try:
        b = args[2]
    except:
        b = [1, 1]

    phi = eig_fun_1(x, par, l, b)
    psi = eig_fun_2(x, par, l, b)

    return np.dot(phi, phi.conjugate()) + np.dot(psi, psi.conjugate())

def eig_fun_mul_1(x, *args):
    
    import numpy as np
    
    par = args[0]
        
    if hasattr(args[1], '__iter__'):
        l = args[1]
    else:
        l = [args[1]]*2
    
    try:
        b = args[2]
    except:
        b = [1, 1]

    phi = eig_fun_1(x, par, l[0], b)
    psi = eig_fun_2(x, par, l[0], b)
    
    phi_star = eig_fun_adj_1(x, par, l[1], b)
    psi_star = eig_fun_adj_2(x, par, l[1], b)

    return np.dot(phi, phi_star.conjugate()) + np.dot(psi, psi_star.conjugate())

def eig_fun_mul_2(x, *args):

    import numpy as np
    
    par = args[0]
    l = args[1]
    
    try:
        b = args[2]
    except:
        b = 1

    z_1 = arbit_fun_1(x, par)
    z_2 = arbit_fun_2(x, par)
    
    phi_star = eig_fun_adj_1(x, par, l, b)
    psi_star = eig_fun_adj_2(x, par, l, b)

    return np.dot(z_1, phi_star.conjugate()) + np.dot(z_2, psi_star.conjugate())

def eig_fun_mul_3(x, *args):

    import numpy as np
    
    par = args[0]
    l = args[1]
    
    try:
        b = args[2]
    except:
        b = 1

    z_1 = arbit_fun_1(x, par)
    z_2 = arbit_fun_2(x, par)
    
    phi = eig_fun_1(x, par, l, b)
    psi = eig_fun_2(x, par, l, b)

    return np.dot(z_1, phi.conjugate()) + np.dot(z_2, psi.conjugate())

def q_ricatti_fun_mul(x, *args):
    
    import numpy as np

    par = args[0]
    l_n, l_m = args[1]
    b_n, b_m = args[2]

    z_1 = q_fun_1(x, par)
    z_2 = q_fun_2(x, par)
    
    phi_n = eig_fun_1(x, par, l_n, b_n)
    phi_m = eig_fun_1(x, par, l_m, b_m)

    psi_n = eig_fun_2(x, par, l_n, b_n)
    psi_m = eig_fun_2(x, par, l_m, b_m)

    return np.dot(z_1 * phi_n, phi_m.conjugate()) + np.dot(z_2 * psi_n, psi_m.conjugate())