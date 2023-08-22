input_variables = ['u', 'v', 't', 'phis', 
             'frland','frlandice','frlake','frocean', # 'frseaice',
             'sphu','qitot','qltot','delp','ps_dyn',
              'dudtdyn', 'dvdtdyn', 'dtdtdyn']
pred_variables = ['u', 'v', 't']

# Path :
data_path = '/Users/vmarchai/Documents/ML_DATA/c48_XY_only_train'
graph_path =  "Output/fullphys_0/Graph"
output_path = 'Output/fullphys_0/'

# PREPROCESSORS :
from src.preprocess.preprocess import Zero_One, Normalizer, Level_Normalizer, Log_Level_Normalizer, Rescaler


preprocess_X_path = f'{output_path}/X_fullphys_preprocess.pickle'
preprocess_Y_path = f'{output_path}/Y_fullphys_preprocess.pickle'

preprocess_X = {
## Surface :
'frlake'    : Zero_One(),
'frland'    : Zero_One(),
#'frlandice' : Zero_One(),
'frocean'   : Zero_One(),
'frseaice'  : Zero_One(),
## Colonnes Physics :
'u'        : Level_Normalizer(normalisation_method = 'std'),
'v'        : Level_Normalizer(normalisation_method = 'std'),
't'        : Level_Normalizer(normalisation_method = 'std'),
'delp'     : Level_Normalizer(normalisation_method = 'std'),
## Clouds
'qitot'        : Zero_One(),
'sphu'        :  Zero_One(),
'qltot'        : Zero_One(),
## Dt dyn
 'dudtdyn':Rescaler(normalisation_method='std'),
 'dvdtdyn':Rescaler(normalisation_method='std'),
 'dtdtdyn':Rescaler(normalisation_method='std')
}

## Prep y
preprocess_Y = {
    'u':Rescaler(normalisation_method='std'),
    'v':Rescaler(normalisation_method='std'),
    't':Rescaler(normalisation_method='std')
}


# Additional params :
test = False
nb_portion = 1
save = True
