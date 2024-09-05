def save_dataframe_to_csv(df, filename, metadata=None, parent_dir=None, process=True):
    """
    Save a DataFrame to a CSV file, optionally with metadata in the first line.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        filename (str): The name of the output CSV file.
        parent_dir (str, optional): The parent directory to save the file in. If None, saves in the current directory.
        metadata (str, optional): Metadata to be stored in the first line of the CSV file as a comment.

    Returns:
        None
    """
    import os
    import pandas as pd
    from .generate_unique_file_path import generate_unique_file_path
    from .cluster_dataframe import cluster_dataframe

    # Ensure the filename ends with '.csv'
    if not filename.endswith('.csv'):
        filename += '.csv'

    if parent_dir:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_filepath = os.path.join(parent_dir, filename)
    else:
        output_filepath = filename

    output_filepath = generate_unique_file_path(output_filepath) # Prevents overwriting an existing file
    
    if process:
        df = cluster_dataframe(df)
        
    if metadata is None:
        df.to_csv(output_filepath, index=False)
        print(f"DataFrame saved to {output_filepath}")
    else:
        with open(output_filepath, 'w') as f:
            metadata_str = str(metadata)
            formatted_metadata_str = metadata_str.replace(',', ';')
            f.write(f"# {formatted_metadata_str}\n")
            df.to_csv(f, index=False)
        print(
            f"DataFrame with metadata '{metadata}' saved to {output_filepath}")