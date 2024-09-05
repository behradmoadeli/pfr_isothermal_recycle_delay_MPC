def generate_unique_file_path(file_path):
    """
    Generate a unique file path by appending a counter to the filename if the file already exists.

    Args:
        file_path (str): The desired file path.

    Returns:
        str: A unique file path that doesn't already exist.
    """
    import os
    
    base_path, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base_path}_{counter}{ext}"
        counter += 1
    return new_file_path