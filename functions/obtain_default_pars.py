def obtain_default_pars(csv_path):
    """
    Create a default parameters dictionary from a CSV file.

    Parameters:
        csv_path (str): The path to the CSV file.

    Returns:
        default_pars: A dictionary containing default parameters set
    """
    import pandas as pd
    
    default_pars = {
        'k': 10,
        'D': 0.1,
        'v': 0.5,
        'tau': 1,
        'R': 0.9,
        'label': 'default'
    }

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_path).fillna('')
        par = df[df['label'] == 'default'].to_dict(orient='records')[0] if any(df['label'] == 'default') else default_pars
    except:
        par = default_pars

    return par