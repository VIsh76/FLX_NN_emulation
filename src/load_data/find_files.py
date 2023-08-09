import warnings
import datetime
import os
from collections import OrderedDict

def get_all_nc4_files_old(input_path):
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


def get_all_nc4_files(data_path, struct_list):
    """
    Args:
        data_path (string): path where all the nc4 files are
        struct_list (list of string): identifier name for each folder containing nc4
    Returns:
        dict: a dict with struct_list elements as keys the output is a list of file path
    """
    struct_list.sort()
    D = OrderedDict()
    for s in struct_list:
        D[s] = []
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
                D[struct_list[ide]].append(root+'/'+file)
    # Check DATA coherence
    for i, key in enumerate(D):
        if i==0:
            ide=len(D[key])
        elif ide != len(D[key]):
            warnings.warn(f'{key} doesnt have the same length / folder integrity wrong')
    return D
