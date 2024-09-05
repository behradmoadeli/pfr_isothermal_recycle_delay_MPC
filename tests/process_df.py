import pandas as pd

# Your original DataFrame
data = {
    'Sol_r': [39.55758771, 6.699931445, 30.63207771, 31.82573541, 39.23081816, 29.53739656, 22.72820463, 18.54827813, 13.63887863],
    'Sol_i': [40.3664694, 31.33657799, 12.93078841, 8.757157638, 42.80403378, 8.676187174, 2.219175649, 32.81860736, 12.79430246],
    'label': [0, 1, 2, 2, 0, 2, 0, 1, 2],
    'error': [0.94597986, 0.232685857, 0.053405559, 0.535112101, 0.213383948, 0.27598515, 0.526382582, 0.028987515, 0.029863001]
}

df = pd.DataFrame(data)
df = df.sort_values(by=['label'])
print(df)

# Group by 'label' and calculate the count of omitted rows for each label
omitted_counts = df.groupby('label').size()

# Creating a DataFrame from the Series
label_df = pd.DataFrame({'instances': omitted_counts})
print(label_df)


# Filter the DataFrame
filtered_df = df.loc[df.groupby('label')['error'].idxmin()]
print(filtered_df)


# Merging the DataFrames
merged_df = pd.merge(filtered_df, label_df, on='label')

# Display the merged DataFrame
print(merged_df)

# Merge the omitted counts back to the filtered DataFrame
# filtered_df = pd.merge(filtered_df, omitted_counts, left_on='label', right_index=True, how='left')

# Display the filtered DataFrame
# print(filtered_df)
