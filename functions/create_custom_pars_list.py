def create_custom_pars_list(csv_path, default_pars=None):
    """
    Create a list of custom parameter dictionaries from a CSV file.

    *Does not return default parameters set!*

    Parameters:
        csv_path (str): The path to the CSV file.

    Returns:
        custom_pars_list: A list of dictionaries containing non-default parameter sets.
    """
    import pandas as pd
    from .obtain_default_pars import obtain_default_pars
    from .create_label import create_label

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path).fillna('')

    # Initialize a list to store dictionaries
    custom_pars_list = []

    if not default_pars:
        default_pars = obtain_default_pars(csv_path)
    
    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        custom_pars = default_pars.copy()

        # Update custom_pars with values from the current row
        for key in default_pars.keys():
            if key in row:
                custom_pars[key] = row[key]

        # Append the customized parameter dictionary to the list
        if custom_pars != default_pars:
            custom_pars['label'] = create_label(custom_pars, default_pars=default_pars)
            custom_pars_list.append(custom_pars)

    return custom_pars_list