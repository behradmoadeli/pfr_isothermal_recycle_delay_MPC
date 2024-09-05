def create_label(par, **kwargs):
    """
    Generates customized labels for parameters.

    Args:
        par (dict): A dictionary containing parameter values. Keys correspond to parameter names,
                    and values correspond to parameter values.

        **kwargs: Additional keyword arguments that correspond to single parameter values to be updated:

            - default_pars (dict, optional): A dictionary containing default parameter values. If not provided,
                                             it attempts to load default parameters from 'pars_list.csv'.
            - pars_list_path (str): Path to a .csv file containing default parameter


    Returns:
        label (str): A label generated based on differing parameter values.
    """
    
    import numpy as np
    from .obtain_default_pars import obtain_default_pars
    
    label = par['label']

    # Create a label based on differing parameters
    if label == '':
        # Load default parameters
        if 'default_pars' in kwargs:
            default_pars = kwargs['default_pars']
        else:
            default_pars = obtain_default_pars(kwargs.get('pars_list_path', 'pars_list.csv'))
        
        # Create new label
        differing_pairs = {}
        for key, value in par.items():
            if key != 'label':
                if not np.isclose(value, default_pars[key]):
                    differing_pairs[key] = value
        label = '_'.join(
            [f"({key}_{value:.3g})" for key, value in differing_pairs.items()])

    return label