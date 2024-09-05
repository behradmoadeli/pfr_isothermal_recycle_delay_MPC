def char_eq_dual(x, *args):
    
    from .char_eq import char_eq
    from .char_eq_adj import char_eq_adj

    y = char_eq(x, *args)
    y_star = char_eq_adj(x, *args)
    
    y_sum = y + y_star

    return y_sum