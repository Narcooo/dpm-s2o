import logging
import os
import datetime
import tqdm
from core.runcontroler import get_env_local_rank

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_logger(name, work_dir, log_level=logging.INFO, file_mode='w', screen=True):

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    os.makedirs((os.path.join(work_dir, 'log')), exist_ok=True)
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M')[2:]
    log_file = os.path.join(work_dir, 'log/{}.txt'.format(name + '_' + date))
    fh = logging.FileHandler(log_file, mode=file_mode)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

# def init_logger():
#     pass