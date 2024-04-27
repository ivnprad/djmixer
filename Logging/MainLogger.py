from multiprocessing import log_to_stderr
from multiprocessing_logging import install_mp_handler
from Logging.Logmodule import ColoredFormatter
from logging.handlers import TimedRotatingFileHandler
import logging

class SingletonLogger:
    _instance = None

    def __new__(cls, logFileName="AIDJ.log"):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
            cls._instance.SetupLogger(logFileName)
        return cls._instance
    
    def SetupLogger(self, logFileName):
        log_to_stderr()  
        self.logger = logging.getLogger("AIDJ")
        self.logger.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(ColoredFormatter())
        self.logger.addHandler(consoleHandler)
        fileHandler = TimedRotatingFileHandler(logFileName, when="midnight", interval=1, backupCount=7)
        fileFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        fileHandler.setFormatter(fileFormatter)
        self.logger.addHandler(fileHandler)   
        install_mp_handler(self.logger)  

    def debug(self, message):
        self.logger.debug("AIDJ " + message)

    def error(self, message):
        self.logger.error("AIDJ " + message)

    def info(self, message):
        self.logger.info("AIDJ " + message)

mainLogger = SingletonLogger()  # This will always return the same logger instance