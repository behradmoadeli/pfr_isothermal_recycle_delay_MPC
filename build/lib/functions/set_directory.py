def set_directory(script_directory, optional_path=None):
    """
    Change the current working directory to the directory where the script is located
    or to a subdirectory within it, if an optional path is provided.

    :param optional_path: An optional path to a subdirectory within the script's directory.
                         If provided, the current working directory will be set to this subdirectory.
    :return: The new current working directory after the change.
    """
    # Get the directory where the current script is located
    import os

    script_directory = os.path.dirname(os.path.abspath(script_directory))

    if optional_path:
        # If optional_path is provided, join it with the script_directory
        target_directory = os.path.join(script_directory, optional_path)
        os.chdir(target_directory)
    else:
        # Change the current working directory to the script's directory
        os.chdir(script_directory)

    # Return the new current working directory
    return os.getcwd()