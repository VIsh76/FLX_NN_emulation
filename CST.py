class CST(object):
    _model_folder="TrainedModels"
    _data_folder='Data'
    _log_folder='Logs'
    _lev = 72

    @staticmethod
    def Model_folder(cls):
        return cls._model_folder

    @staticmethod
    def Data_folder(cls):
        return cls._data_folder

    @staticmethod
    def Log_folder(cls):
        return cls._log_folder

    @staticmethod
    def lev(cls):
        return cls._lev
