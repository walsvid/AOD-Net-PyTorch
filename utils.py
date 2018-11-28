import argparse
import logging
from functools import wraps

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LEVEL = logging.DEBUG
logging.basicConfig(format=FORMAT, level=LEVEL)
log = logging.getLogger(__name__)
log.info('Entered module: %s' % __name__)


def logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(fn.__name__)
        log.info('Start running %s' % fn.__name__)

        out = fn(*args, **kwargs)

        log.info('Done running %s' % fn.__name__)
        # Return the return value
        return out

    return wrapper


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')