from datetime import datetime 
import json
data = json.load(open("general_log 2021-11-16 20_16_48.json"))

hiro_time = 0
human_time = 0
total_cycles = len(list(data.keys()))
last_time = None
for iteration in data.keys():
    
    start_time = datetime.strptime(data[iteration]["start"]["timestamp"], "%Y-%m-%d %H:%M:%S")
    move_time = datetime.strptime(data[iteration]["moves"]["timestamp"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(data[iteration]["end"]["timestamp"], "%Y-%m-%d %H:%M:%S")

    hiro_elapsed_time = end_time - start_time
    if last_time is not None:
        human_elapsed_time = start_time - last_time
        human_time += human_elapsed_time.total_seconds()
    last_time = end_time

    hiro_time += hiro_elapsed_time.total_seconds()

print(total_cycles)
print(hiro_time)
print(human_time)
    



