def plot_single_df(
        csv_path, filter=False,
        real_lower_bound=None, real_upper_bound=None, imag_lower_bound=None, imag_upper_bound=None
):
    """
    Plot a DataFrame based on several inputs.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title for the plot.
        metadata (dict): Dictionary containing metadata information.
        filter_real (float): Minimum value for real part.
        filter_imag (float): Absolute maximum value for imaginary part.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from .load_dataframe_from_csv import load_dataframe_from_csv

    df, label, metadata = load_dataframe_from_csv(csv_path)

    if filter:
        # Modify default args
        if not real_lower_bound:
            real_lower_bound = -np.inf
        if not real_upper_bound:
            real_upper_bound = np.inf
        if not imag_lower_bound:
            imag_lower_bound = -np.inf
        if not imag_upper_bound:
            imag_upper_bound = np.inf
        # Filter rows based on criteria
        filtered_df = df[
            (df['Sol_r'] > real_lower_bound) &
            (df['Sol_r'] < real_upper_bound) &
            (df['Sol_i'] > imag_lower_bound) &
            (df['Sol_i'] < imag_upper_bound)
        ]
    else:
        filtered_df = df

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    ax.scatter(filtered_df['Sol_r'], filtered_df['Sol_i'])
    ax.set_title(label)
    ax.set_xlabel('Sol_r')
    ax.set_ylabel('Sol_i')

    # Set axis limits to ensure visibility of x=0 and y=0 lines
    ax.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0
    ax.grid(True)
    plt.show()
    print(metadata)

    return df, label, metadata
