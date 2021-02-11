import os
import yaml
import inspect
import functools

from yaml import Dumper, Loader

# ######################################################################################################################
#                                                       CONFIGURABLE
# ######################################################################################################################
class Configurable:
    """A utility class for creating configurable models.
    A class can have
        - parameters
        - hyperparameters
    """

    # =============================================== VALIDATE INTERNAL ================================================
    def _validate_params(self):
        """
        Utility function for validating the parameters configuration. It is supposed that the parameters are
        already set in self.
        Example:
            assert self.param > 0, "Param not ok"
        """
        pass

    # =============================================== VALIDATE EXTERNAL ================================================
    def _validate_hparams(self, kwargs):
        """
         Utility function for validating the hyper parameters configuration. It is supposed that the hyper parameters
         configuration is set. Its attribute can be accesed via the kwargs arguments.
         Example:
            assert kwargs['hparamName'] > 0, 'Param not ok.Param is bad'
        :param kwargs: the hParams configuration dictionary
        """

        pass

    # =============================================== BUILD ============================================================
    def build(self):
        raise NotImplementedError("Please initialize \"heavy\" objects here")

    # =============================================== EXTERNAL CONFIG ==================================================
    def build_params(self, validate=True):
        """
        Parameters are fields that are passed via the __init__ method in the subclass as named arguments (both
        positional and keywords). They also have to be set in the self object having the same name as in the contructor
        signature.
        Example:
            def __init__(self, a b, d= 3, *args, **kwargs)
                self.a = a
                self.anotherNameForB = b
                self.d = d
            Only <a> and <d> will be considered parameters, as they are named parameters, and
            they are both set in the method signature and set into the object
        As a side effect, calling this function will create a list of parameters will keep the name of these
        parametrs. This list will be called when saving the configuration.

        """

        assert not hasattr(self, 'paramsAttrs'), "Please do not use [paramsAttrs] field."

        paramsAttrs = []
        initSig = inspect.signature(self.__init__)

        for param in initSig.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                if hasattr(self, param.name):
                    paramsAttrs.append(param.name)

        if validate:
            self._validate_params()

        setattr(self, '__paramsAttrs', paramsAttrs)

    # =============================================== INTERNAL CONFIG ==================================================
    @staticmethod
    def hyper_params(validate=True):

        def actual_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = func(*args, **kwargs)

                # args[0] refferes to self

                for key in res.keys():
                    res[key] = kwargs.get(key, res[key])
                    setattr(args[0], key, res[key])

                if validate:
                    args[0]._validate_hparams(res)

                setattr(args[0], '__hParamsAttrs', [str(key) for key in res.keys()])

                return res

            return wrapper

        return actual_decorator

    # =============================================== SAVE CONFIG ======================================================
    def save_config(self, path, hParams=True, params=True):
        """
        Utility method for saving the configuration dictionaries.

        :param hParams: path for saving the hyper parameters config
        :param params: path for saving the parameters config
        """

        saveDict = {}

        if hParams:
            assert hasattr(self, '__hParamsAttrs'), 'Can not find hParams config'
            saveDict['hParams'] = self.get_config_dict('hParams', False, True)

        if params:
            assert hasattr(self, '__paramsAttrs'), 'Can not find params config'
            saveDict['params'] = self.get_config_dict('params', False, True)

        # get the directory and create it if it does not exists
        pathDir = os.path.split(path)[0]

        if not os.path.exists(pathDir) and pathDir is not '':
            os.makedirs(pathDir)

        # save the configuration
        with open(path, 'w') as f:
            yaml.dump(saveDict, f, Dumper)

    # =============================================== LOAD CONFIG ======================================================
    def load_config(self, path, hParams=True, params=True, loadExisting=True, validate=True):
        """
        Utility function for loading a desired configuration.

        :param hParams: path for loading the hParams config
        :param params: path for loading the params config
        :param loadExisting: use the existing keys in list or use the loaded keys
        :return:
        """

        assert os.path.exists(path), 'The config file could not be found: %s' % path

        with open(path, 'r') as f:
            myDict = yaml.load(f, Loader)

        if hParams: assert 'hParams' in myDict, 'Could not find hyper params  into dict'
        if params: assert 'params' in myDict, 'Could not find params into dict'

        # only load the keys existing in the actual configuration
        if loadExisting:
            if hParams:
                assert hasattr(self, '__hParamsAttrs'), 'The object does not have hParams config set.'

                if validate:
                    self._validate_hparams(myDict['hParams'])

                for attr in getattr(self, '__hParamsAttrs'):
                    if attr in myDict['hParams']:
                        setattr(self, attr, myDict['hParams'][attr])
                    else:
                        raise ValueError('Could not load attr %s from dict with keys: %s' % \
                                         (attr, ['%s , ' % str(key) for key in myDict['hParams'].keys()]))

            if params:
                assert hasattr(self, '__paramsAttrs'), 'The object does not have params config set.'

                for attr in getattr(self, '__paramsAttrs'):
                    if attr in myDict['params']:
                        setattr(self, attr, myDict['params'][attr])
                    else:
                        print('Warning: ignored ', attr, 'key')
                        # raise ValueError('Could not load attr %s from dict with keys: %s' % \
                        #                  (attr, ['%s , ' % str(key) for key in myDict['params'].keys()]))

                if validate:
                    self._validate_params()

        # load the all the keys that are in the configuration value
        else:
            if hParams:
                if validate:
                    self._validate_hparams(myDict)

                setattr(self, '__hParamsAttrs', [str(key) for key in myDict.keys()])
                for key in myDict['hParams'].keys():
                    setattr(self, key, myDict[key])

            if params:
                setattr(self, '__paramsAttrs', [str(key) for key in myDict.keys()])
                for key in myDict['params'].keys():
                    setattr(self, key, myDict[key])

                if validate:
                    self._validate_params()

        return self

    # =============================================== GET CONFIG DICT ==================================================
    def get_config_dict(self, name, returnMissing=False, includeMisssing=True):
        """
        Build and get the configuration dictionary.

        :param name: [hParams/params]
        :param returnMissing: flag for returning the missing attributes
        :return a dict for the specified configuration
        """

        # validate parameters
        cfgName = '__' + name + 'Attrs'
        assert name in ['hParams', 'params'], 'Please choose from [hParams/params]. Currently [%s]' % name
        assert hasattr(self, cfgName), 'The object does not have this config set: %s' % cfgName

        configDict = {}
        missingAttrs = []

        for attr in getattr(self, '__' + name + 'Attrs'):
            # if the object does not have the attribute, store it
            if hasattr(self, attr):
                configDict[attr] = self.__dict__[attr]
            else:
                if includeMisssing:
                    configDict[attr] = None
                missingAttrs.append(attr)

        # return the desired parameters
        if returnMissing:
            return configDict, missingAttrs

        return configDict

    # =============================================== PRINT CONFIG =====================================================
    def print_config(self):
        print('============= CONFIG  =============')
        print('------------- PARAMS  ------------')
        print(yaml.dump(self.get_config_dict('hParams'), default_flow_style=False))
        print('------------- HPARAMS ------------')
        print(yaml.dump(self.get_config_dict('params'), default_flow_style=False))


# =============================================== TEST =================================================================
if __name__ == '__main__':
    class A(Configurable):
        def __init__(self, a, b, c, d=5, **kwargs):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.build_params()

            self.mehe = 123
            self.build_hparams(**kwargs)

        @Configurable.hyper_params()
        def build_hparams(self, **kwargs):
            return {'aha': 3, 'heeh': 'sad'}

        def _validate_hparams(self, kwargs):
            assert kwargs['aha'] > 2, 'tztz'


    a = A(1, 2, 3, 4, hohoho=5, heeh=11212)

    a.save_config('path.yaml')
    a.load_config('path.yaml')
    a.print_config()
