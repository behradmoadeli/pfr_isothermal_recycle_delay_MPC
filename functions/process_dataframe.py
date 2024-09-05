def process_dataframe(df_raw, n):
    """
    Process a pandas DataFrame by adding an 'instances' column, rounding values,
    and handling duplicates. Also replaces small 'Sol_i' values with zero.

    Args:
        df (pandas.DataFrame): The input DataFrame with 'Sol_r' and 'Sol_i' columns.

    Returns:
        pandas.DataFrame: Processed DataFrame with 'instances' column updated.
    """
    import pandas as pd
    import numpy as np
    from .significant_digits import significant_digits

    # Filter non-solution data
    df = df_raw[df_raw['ier']==1]
    
    # Add 'instances' column with all values set to 1
    df['instances'] = 1
    
    # Replace values in 'Sol_i' column with 0 if absolute value is less than 1e-8
    df['Sol_i'] = np.where(np.abs(df['Sol_i']) < 1e-3, 0, df['Sol_i'])
    
    # Iterate through rows and perform rounding and duplicate handling
    for index, row in df.iterrows():
        # Round 'Sol_r' and 'Sol_i' columns by n digits
        df.at[index, 'Sol_r'] = round(row['Sol_r'], significant_digits(row['Sol_r'],n))
        df.at[index, 'Sol_i'] = round(row['Sol_i'], significant_digits(row['Sol_i'],n))
        
    for index, row in df.iterrows():
        # Check for duplicates and update instances
        duplicates = df[(df['Sol_r'] == row['Sol_r']) & (df['Sol_i'] == row['Sol_i'])]
        if len(duplicates) > 1:
            df.at[duplicates.index[0], 'instances'] += len(duplicates) - 1
            df.drop(duplicates.index[1:], inplace=True)
    
    return df