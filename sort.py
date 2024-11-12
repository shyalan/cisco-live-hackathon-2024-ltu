import datetime
import os

def parse_time_entry(line):
    parts = line.strip().split(', ')
    if len(parts) != 3:
        return None
    try:
        start_time = datetime.datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.datetime.strptime(parts[1], '%Y-%m-%d %H:%M:%S')
        duration = float(parts[2].split()[0])
        return start_time, end_time, duration
    except Exception as e:
        print(f"Error parsing line: {line}, Error: {e}")
        return None

def read_raw_file():
    with open('raw_held_log.txt', 'r') as f:
        lines = f.readlines()
    entries = [parse_time_entry(line) for line in lines]
    return [e for e in entries if e]

def get_last_entry_id():
    if not os.path.exists('held_log.txt'):
        return 0
    with open('held_log.txt', 'r') as f:
        lines = f.readlines()
    if not lines:
        return 0
    last_line = lines[-1]
    last_id = last_line.split(',')[0].strip()
    return int(last_id[2:])

def write_detection_log(entry_id, start_time, end_time, total_duration):
    if total_duration >= 1.0:
        with open('held_log.txt', 'a') as f:
            f.write(f"{entry_id}, {start_time}, {end_time}, {total_duration:.2f} seconds\n")

def update_raw_file(entries):
    with open('raw_held_log.txt', 'w') as f:
        for entry in entries:
            start, end, duration = entry
            f.write(f"{start}, {end}, {duration:.2f} seconds\n")

def process_entries(entries):
    processed = []
    current_start = None
    current_end = None
    total_duration = 0
    last_entry_id = get_last_entry_id()
    entry_counter = last_entry_id + 1
    i = 0

    while i < len(entries):
        start, end, duration = entries[i]
        if current_start is None:
            current_start = start
            current_end = end
            total_duration = duration
            entry_id = f"DN{entry_counter:04d}"
        else:
            time_difference = (start - current_end).total_seconds()
            if time_difference <= 3:
                current_end = end
                total_duration += duration
            else:
                write_detection_log(entry_id, current_start, current_end, total_duration)
                entry_counter += 1
                current_start = start
                current_end = end
                total_duration = duration
                entry_id = f"DN{entry_counter:04d}"
        
        processed.append((start, end, duration))
        i += 1

    if current_start is not None:
        write_detection_log(entry_id, current_start, current_end, total_duration)

    remaining_entries = [e for e in entries if e not in processed]
    update_raw_file(remaining_entries)

if __name__ == "__main__":
    entries = read_raw_file()
    if entries:
        process_entries(entries)
    else:
        print("No valid entries found in raw_detection_log.txt.")
