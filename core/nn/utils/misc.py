import os
import time
import logging

# ================================================== GET LOGGET ========================================================
def create_logger(name, saveDir, logLevel):

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
