import numpy as np

class DummyTest():
    def __init__(self) -> None:
        self._validated = False
        self.logs = dict()

    @property
    def validated(self):
        return self._validated
    
    def perform(self):
        return True
    
class EqualTest(DummyTest):
    def __init__(self) -> None:
        super().__init__()
    
    def perform(self,data_1, data_2, strict=False, vars=None):
        passed = True
        if vars is None:
            vars = data_1.keys()
        for var in data_1:
            if var in data_2:
                delta = np.max(abs(data_1[var].data - data_2[var].data))
                passed = delta <= 0
                self.logs[var] = delta 
            else:
                if strict:
                    passed=False
                self.logs[var] = -1
        self._validated = passed

class NonZeroTest(DummyTest):
    def __init__(self) -> None:
        super().__init__()
    
    def perform(self, data_1, strict=False, vars=None, **kwarg):
        passed = True
        if vars is None:
            vars = data_1.keys()
        for var in data_1:
            if var in data_1:
                delta = np.max(abs(data_1[var].data))
                passed = delta <= 0
                self.logs[var] = delta 
            else:
                if strict:
                    passed=False
                self.logs[var] = -1
        self._validated = passed
