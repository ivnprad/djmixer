import logging
from logging.handlers import TimedRotatingFileHandler
import datetime

# ANSI escape sequences for colors
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[32;1m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    custom_format = "%(asctime)s - %(levelname)s - %(message)s"

    COLORS = {
        logging.DEBUG: green,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: red,
    }
    
    def format(self, record):
        log_fmt = self.COLORS.get(record.levelno) + self.custom_format + self.reset
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
    
def SetupLogging(logFileName = "AIDJ.log"):

    # Setup logging
    logger = logging.getLogger("AIDJ")
    logger.setLevel(logging.DEBUG)

    # Console handler with colored output
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(ColoredFormatter())
    logger.addHandler(consoleHandler)

    #TODO File handler with standard formatting
    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #logFileName = f"aidj_{timestamp}.log"
    #logFileName = f"Aidj.log"
    fileHandler = TimedRotatingFileHandler(logFileName, when="midnight", interval=1, backupCount=7)
    fileFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fileHandler.setFormatter(fileFormatter)
    logger.addHandler(fileHandler)

    return logger
