import warnings
import datetime
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


def get_all_nc4_files_fullphys(data_path, D):
    """
    = 20000414_2107z
    Given the path of DATA/
    Requires the main folder to have 3 subfolder : 
    - ML_DATA_input
    - ML_DATA_nophys
    - ML_DATA_phys    
    Dates of the input are heartbeat minutes lower than output
    INPUT have a date that is 
    Will provide path to : 
    - input 
    - output no phys
    - output phys

    Args:
        input_path (string): path where all the nc4 files are

    Returns:
        list of tuple: a list of each tuple (input_file_path, output_file_path)
    """
    keys = list(D.keys())
    keys.sort()
    ide=-1
    files = []
    for root, dirs, files in os.walk(data_path):
        dirs.sort()
        files.sort()
        flag = False
        for file in files:
            # choose only input files and nc4 files
            if file.endswith(".nc4"):
                if not flag:
                    flag = True
                    ide+=1
                D[keys[ide]].append(root+'/'+file)
    return D
