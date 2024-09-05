import pandas as pd

# Assuming your DataFrame is named df
data = {
    'Sol_r': [-69.50153856, -66.40610503, -0.696307024, 8.664853645, -53.01230805, 2.368915736, -57.6762209, -31.83310546, 6.417267919, -6.597022697, -2.109798105],
    'Sol_i': [12.14894894, 28.22419049, 14.8315016, 0, 14.91664799, 38.17483882, 3.257372209, 22.28046786, 14.14369992, 11.09790937, 28.65010905],
    'cmplx_cnj': ['Yes', 'Yes', 'Yes', 'N_A', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}


df = pd.DataFrame(data)
print(df.sort_values(by=['Sol_r']))

# Function to re-apply complex conjugate values
def update_dataframe(row):
    if row['cmplx_cnj'] != 'N_A':
        # Create a new row with identical values
        new_row = row.copy()
        # Set Sol_i to the negative of its current value if cmplx_cnj is 'Yes'
        if row['cmplx_cnj'] == 'Yes':
            new_row['cmplx_cnj'] = 'No'
        else:
            new_row['cmplx_cnj'] = 'Yes'
        new_row['Sol_i'] = -row['Sol_i']
        # Append the new row to the DataFrame
        df.loc[len(df.index)] = new_row

# Apply the function to each row of the DataFrame
df.apply(update_dataframe, axis=1)

# Reset index to make it continuous
df.reset_index(drop=True, inplace=True)

# Display the updated DataFrame
print(df.sort_values(by=['Sol_r']))