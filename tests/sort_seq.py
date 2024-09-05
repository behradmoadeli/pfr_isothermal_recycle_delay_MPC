sort_sequence = ['v', 'D', 'k', 't', 'R']

def is_digit(input_str):
    # Check if the input is an empty string or a decimal point
    if input_str == '' or input_str == '.'or input_str == '-':
        return True
    
    try:
        # Try to convert the input to a float
        float_value = float(input_str)
        return True
    except ValueError:
        # If the conversion to float fails, return False
        return False

def custom_sort_key(item):
    second_element = str(item[1])  # Convert the second element to a string
    first_alpha_letter = next(
        (char for char in second_element if char.isalpha()), '')
    l = sort_sequence.index(first_alpha_letter) if first_alpha_letter in sort_sequence else len(sort_sequence)
    
    first_alpha_letter_value = ''
    for char in second_element:
        if is_digit(char):
            first_alpha_letter_value += char
        if not is_digit(first_alpha_letter_value):
            break
    
    v = float(first_alpha_letter_value)
            
    return l * 1e6 + v
    

custom_sort_key(['a', '(D_0.05)'])


results = [
    ['item1', '(k_-3)'],
    ['item2', '(k_-2)'],
    ['item3', '(k_1)'],
    ['item5', 'k_2'],
]


# Sort the results list based on the custom sorting key
results = sorted(results, key=custom_sort_key)



for i in results:
    print(i)