def char_eq_dual(x, *args):
    
    import numpy as np
    from .char_eq import char_eq
    from .char_eq_adj import char_eq_adj

    y = char_eq(x, *args)
    y_star = char_eq_adj(x, *args)
    
    y_sum = np.sqrt(y*y.conjugate() + y_star*y_star.conjugate())

    return y_sum