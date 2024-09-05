def plot_multiple_datasets(
        parent_directory, save_path=None,
        filter=False, filter_range=[(-200, 10), 100],
        rows=None, cols=None, figsize=None, **kwargs):
    """
    Plot multiple datasets from CSV files in a grid layout.

    Args:
        parent_directory (str): The directory containing CSV files to plot.
        rows (int, optional): Number of rows in the grid. If None, calculated based on dataset count.
        cols (int, optional): Number of columns in the grid. If None, calculated based on dataset count.
        figsize (tuple, optional): Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to matplotlib.subplots().

    Returns:
        None
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from .load_dataframe_from_csv import load_dataframe_from_csv
    from .generate_unique_file_path import generate_unique_file_path

    # List all CSV files in the parent directory
    csv_files = [file for file in os.listdir(
        parent_directory) if file.endswith('.csv')]
    results = []

    # Load dataframes from CSV files and categorize into 'default' and other datasets
    for csv_file in csv_files:
        csv_path = os.path.join(parent_directory, csv_file)
        df, base_filename, metadata = load_dataframe_from_csv(csv_path)
        if base_filename == 'default':
            default = df
        else:
            results.append((df, base_filename, metadata))

    sort_sequence = ['v', 'D', 'k', 't', 'R']

    def is_digit(input_str):
        # Check if the input is an empty string or a decimal point
        if input_str == '' or input_str == '.' or input_str == '-':
            return True
        
        try:
            # Try to convert the input to a float
            float_value = float(input_str)
            return True
        except ValueError:
            # If the conversion to float fails, return False
            return False

    # Define a sorting function that extracts the first alphabetic letter from the second element of each tuple
    def custom_sort_key(item):
        second_element = str(item[1])  # Convert the second element to a string
        first_alpha_letter = next(
            (char for char in second_element if char.isalpha()), '')
        l = sort_sequence.index(first_alpha_letter) if first_alpha_letter in sort_sequence else len(sort_sequence)
        
        first_alpha_letter_value = ''
        for char in second_element:
            if is_digit(char):
                first_alpha_letter_value += char
        
        v = float(first_alpha_letter_value)
                
        return l * 1e6 + v
      
    # Sort the results list based on the custom sorting key
    results = sorted(results, key=custom_sort_key)

    # Extract datasets and calculate the number of datasets
    datasets = [result[0] for result in results]
    labels = [result[1] for result in results]
    num_datasets = len(datasets)

    # Calculate rows and columns for the grid layout if not specified
    if rows is None and cols is None:
        rows = int(np.ceil(np.sqrt(num_datasets)))
        cols = int(np.ceil(num_datasets / rows))
    elif rows is None:
        rows = int(np.ceil(num_datasets / cols))
    elif cols is None:
        cols = int(np.ceil(num_datasets / rows))

    mid_row = int(np.ceil(rows/2))

    # Set default figure size if not specified
    if figsize is None:
        figsize = (cols * 8, rows * 8)

    # Create the subplots grid
    fig, axes = plt.subplots(rows+1, cols, figsize=figsize, **kwargs)

    # Flatten the axes array if there's only one dataset
    if num_datasets == 1:
        axes = np.array([[axes]])

    if filter:
        # Filter rows based on criteria
        default = default[
            (default['Sol_r'] > min(filter_range[0])) &
            (default['Sol_r'] < max(filter_range[0])) &
            (default['Sol_i'] > -abs(filter_range[1])) &
            (default['Sol_i'] < abs(filter_range[1]))
        ]

    # Iterate through datasets and plot them on the grid
    for i, dataset in enumerate(datasets):

        if filter:
            # Filter rows based on criteria
            dataset = dataset[
                (dataset['Sol_r'] > min(filter_range[0])) &
                (dataset['Sol_r'] < max(filter_range[0])) &
                (dataset['Sol_i'] > -abs(filter_range[1])) &
                (dataset['Sol_i'] < abs(filter_range[1]))
            ]

        row = i % rows  # Calculate current row index
        col = i // rows  # Calculate current column index

        if rows > 1 and cols > 1:
            index = (row, col)
        else:
            index = (max(row, col),)
        
        # Plot default dataset
        if row == mid_row:
            # Plot each dataset
            axes[*index].scatter(default['Sol_r'], default['Sol_i'])
            axes[*index].set_title('Default')

            # Add gridlines
            axes[*index].grid(True)

            # Show both x-axis and y-axis lines
            # Horizontal line at y=0
            axes[*index].axhline(0, color='black', linewidth=0.8)
            # Vertical line at x=0
            axes[*index].axvline(0, color='black', linewidth=0.8)

        if row >= mid_row:
            if rows > 1 and cols > 1:
                index = (row+1, col)
            else:
                index = (max(row+1, col),)

        # Plot each dataset
        axes[*index].scatter(dataset['Sol_r'], dataset['Sol_i'])
        axes[*index].set_title(labels[i])

        # Add gridlines
        axes[*index].grid(True)

        # Show both x-axis and y-axis lines
        # Horizontal line at y=0
        axes[*index].axhline(0, color='black', linewidth=0.8)
        # Vertical line at x=0
        axes[*index].axvline(0, color='black', linewidth=0.8)

    # Remove empty subplots if there are more grids than datasets
    for i, ax in enumerate(axes.flat):
        if not ax.get_title():
            ax.axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        if not save_path.endswith('.png'):
            save_path += '.png'
        save_path = generate_unique_file_path(save_path)
        plt.savefig(save_path)  # Save the figure as an image file
        plt.close()  # Close the figure to release resources
