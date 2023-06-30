import warnings
import os

def get_all_nc4_files(input_path):
    """Given an input path, will return a list of tuple

    Args:
        input_path (string): path where all the nc4 files are

    Returns:
        list of tuple: a list of each tuple (input_file_path, output_file_path)
    """
    nc4_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # choose only input files and nc4 files
            if file.endswith(".nc4") and "_in." in file:
                file_path = os.path.join(root, file)
                nc4_files.append(file_path)
    L = []
    for input_file in nc4_files:
        output_file = input_file.replace('_in.', '_out.')
        if os.path.exists(output_file):
            L.append(  (input_file, output_file) )
        else:
            warnings.warn(f"{output_file} doesnt exist, {input_file} is not added")
    return L