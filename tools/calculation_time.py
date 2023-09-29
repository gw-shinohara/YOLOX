import time

def calc_time(meat_paste):
    def wrapper(*args, **kwargs):
        st=time.time()
        result = meat_paste(*args, **kwargs)
        ed=time.time()
        dt=ed-st
        print(f"Execution time: {dt}s")
        print(f"Execution FPS : {1/dt}")
        
        return result
    return wrapper