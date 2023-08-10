import xarray as xr
import os
from find_files import get_all_nc4_files


class XY_Generator():
    def __init__(self):
        """Will produce two list of files X,Y for ML training
        From the output of a discovery experiment generating full experiments

        Args:
            input_folder (str): Main folder where the all the geos output are
            output_folder (str): Folder to save X,Y
            struct_list (list of string): identifier name for each folder containing nc4
        """
    def perform(self, input_folder, output_folder, **kwarg):
        pass

    def generate_Y(self, input):
        pass

    def generate_X(self):
        pass

class XY_GEOS_full_output(XY_Generator):
    def __init__(self):
        super().__init__()
        self.required_files = ['nophys_input', 'nophys_output', 'phys_output']

    def generate_Y(self, id):
        data = xr.open_dataset(self.list_of_files['nophys_input'][id])
        return data
 
    def generate_X(self, id):
        data_phy    = xr.open_dataset(self.list_of_files['phys_output'][id])
        data_nophy  = xr.open_dataset(self.list_of_files['nophys_input'][id])
        for v in data_phy:
            data_phy[v] = data_phy[v] - data_nophy[v]
        return data_phy

    def perform(self, input_folder, output_folder, struct_list):
        for f in self.required_files:
            if not f in struct_list:
                print(f'Files {f} is not found in structure')
                assert(False)
        self.list_of_files = get_all_nc4_files(input_folder, struct_list)
        self.file_number = len(self.list_of_files['nophys_input'])



    
    
    
    
    