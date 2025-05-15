import os, sys
import logging
from logging.handlers import TimedRotatingFileHandler
from  logging.handlers import RotatingFileHandler


class WriteLog:
    def __init__(self, file_name, log_name="main"):
        self.filename = file_name
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        self.log_path = os.path.join(os.getcwd(), file_name)

        # 输出到文件
        fh = RotatingFileHandler(filename=self.log_path, maxBytes=4096 * 1024, backupCount=5, encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        if len(self.logger.handlers) < 2:
            self.logger.addHandler(fh)
            # self.logger.addHandler(sh)
        else:
            pass

    def write_in(self, message, level="INFO"):
        if level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def remove(self):
        for hander in self.logger.handlers:
            self.logger.removeHandler(hander)

"""
if __name__ == "__main__":
    print("hello")
    s = WriteLog("log")
"""