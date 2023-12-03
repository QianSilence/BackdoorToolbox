class Log:
    log_path = None
    def __init__(self, log_path = None):
        Log.log_path = log_path
    def __call__(self, msg):
        print(msg, end='\n')
        with open(Log.log_path,'a') as f:
            f.write(msg)
    def set_log_path(self, log_path=None):
        Log.log_path = log_path

log = Log(log_path="")

