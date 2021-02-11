import os
import time
import logging

# ================================================== GET LOGGER ========================================================
def create_logger(name, saveDir, logLevel):
    """
    Initialize a logger. The output will be saved into saveDir.

    :param name:      The name of the logger.
    :param saveDir:   Saving directory for the logging
    :param logLevel:  Log level (see logging library)
    :return:
    """
    head         = '[%(asctime)-15s] : %(message)s'
    timeStr      = time.strftime('%Y-%m-%d-%H-%M')
    logFilePath  = '{}_{}.log'.format(name, timeStr)
    logFilePath  = os.path.join(saveDir, logFilePath)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    logging.basicConfig(filename=str(logFilePath), format=head)

    logger = logging.getLogger()
    logger.setLevel(logLevel)

    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
