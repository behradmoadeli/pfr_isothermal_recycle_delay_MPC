def load_dataframe_from_csv(input_filepath):
    """
    Load a DataFrame from a CSV file, along with metadata if present in the first line.

    Args:
        input_filepath (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
        str: The base filename (without extension) of the input file.
        str or None: Metadata if present, else None.
    """
    import pandas as pd
    import os

    with open(input_filepath, 'r') as f:
        metadata_line = f.readline().strip()
        if metadata_line.startswith('#'):
            metadata = metadata_line.lstrip('# ').strip()
            df = pd.read_csv(input_filepath, comment='#')
        else:
            metadata = None
            df = pd.read_csv(input_filepath)
    
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    return df, base_filename, metadata