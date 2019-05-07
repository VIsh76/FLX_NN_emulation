import psutil

# Contain utilitary functions

def print_ram_usage():
    process = psutil.Process(os.getpid())
    ram_usage = round(process.memory_info().rss/float(2**30), 2)
    print("RAM usage: {}GB".format(ram_usage))
