import datetime

def get_date_and_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')