input_variables = ['u', 'v', 't', 'phis', 
             'frland','frlandice','frlake','frocean', # 'frseaice',
             'sphu','qitot','qltot','delp','ps_dyn',
              'dudtdyn', 'dvdtdyn', 'dtdtdyn']
pred_variables = ['u', 'v']

# Path :
data_path = '/Users/vmarchai/Documents/ML_DATA/c48_XY_only_train'
data_path_test = '/Users/vmarchai/Documents/ML_DATA/c48_XY_only_test'

graph_path =  "Output/fullphys_0/Graph"
output_path = 'Output/fullphys_0'

# PREPROCESSORS :
from src.preprocess.preprocess import Zero_One, Normalizer, Level_Normalizer, Log_Level_Normalizer, Rescaler


preprocess_X_path = f'{output_path}/X_fullphys_preprocess.pickle'
preprocess_Y_path = f'{output_path}/Y_fullphys_preprocess.pickle'

preprocess_X = {
## Surface :
'frlake'    : Zero_One('frlake'),
'frland'    : Zero_One('frland' ),
#'frlandice' : Zero_One(),
'frocean'   : Zero_One('frocean'),
'frseaice'  : Zero_One('frseaice'),
## Colonnes Physics :
'u'        : Level_Normalizer('u', normalisation_method = 'std'),
'v'        : Level_Normalizer('v', normalisation_method = 'std'),
't'        : Level_Normalizer('t', normalisation_method = 'std'),
'delp'     : Level_Normalizer('delp', normalisation_method = 'std'),
## Clouds
'qitot'        : Zero_One('qitot'),
'sphu'        :  Zero_One('sphu'),
'qltot'        : Zero_One('qltot'),
## Dt dyn
 'dudtdyn':Rescaler('dudtdyn', normalisation_method='std'),
 'dvdtdyn':Rescaler('dvdtdyn', normalisation_method='std'),
 'dtdtdyn':Rescaler('dtdtdyn', normalisation_method='std')
}

## Prep y
preprocess_Y = {
    'u':Rescaler('u', normalisation_method='std'),
    'v':Rescaler('v', normalisation_method='std'),
    't':Rescaler('t', normalisation_method='std')
}


# Additional params :
test = False
nb_portion = 1
save = True
