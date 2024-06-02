import logging
import os, sys
from termcolor import colored
import requests, time, datetime

def print_log(print_string, log=None):
    print("{}".format(print_string))
    if log!=None:
        log.write('{}\n'.format(print_string))
        log.flush()

def get_beijin_time():
    try:
        url = 'https://beijing-time.org/'
        request_result = requests.get(url=url)
        if request_result.status_code == 200:
            headers = request_result.headers
            net_date = headers.get("date")
            gmt_time = time.strptime(net_date[5:25], "%d %b %Y %H:%M:%S")
            bj_timestamp = int(time.mktime(gmt_time) + 8 * 60 * 60)
            crt = datetime.datetime.fromtimestamp(bj_timestamp)
            str_crt=str(crt.month)+'-'+str(crt.day)+'-'+str(crt.hour)+'-'+str(crt.minute)
    except Exception as exc:
        crt = datetime.datetime.now()
        str_crt=str(crt.month)+'-'+str(crt.day)+'-'+str(crt.hour+8)+'-'+str(crt.minute)
    return str_crt

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        
        return prefix + " " + log

def get_logger(name='train', output=None, color=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # STDOUT
    stdout_handler = logging.StreamHandler( stream=sys.stdout )
    stdout_handler.setLevel( logging.DEBUG )

    plain_formatter = logging.Formatter( 
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S" )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        formatter = plain_formatter
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # FILE
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            filename = output
        else:
            os.makedirs(output, exist_ok=True)
            filename = os.path.join(output, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger

        
