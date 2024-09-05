def finite_dif_fun_1(phi, dz, dt, par):
    
    import numpy as np

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    y = phi[1] + dt * (
        D /(dz**2) * (
            phi[0] + phi[2] - 2*phi[1]
        ) - v / (2*dz) * (phi[2] - phi[0]
        ) + k * phi[1]
    )

    return y

def finite_dif_fun_2(psi, dz, dt, par):
    
    import numpy as np

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    y = psi[0] + dt * (
        1 /(t*dz) * (psi[1] - psi[0])
    )

    return y

def finite_dif_fun_3(phi_1, psi_0, u, dz, par):
    
    import numpy as np

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    y = (D * phi_1 + v * dz * (
        R * psi_0 + (1-R) * u
    ))/(D + v*dz)

    return y