import re
from typing import List, Iterator
import pandas as pd
# Define the pattern to extract the data from each line
pattern_base = (
    r"(?P<time_ns>\d+)\s+n:(?P<node>\d+)\s+"
    r"(?P<port>\d+):(?P<queue>\d+)\s+"
    r"(?P<queue_length>\d+)\s+"
    r"(?P<event_type>\w+)\s+"
    r"ecn:(?P<ecn>\d+)\s+"
    r"(?P<src_ip_port>0b[0-9a-f]+)\s+"
    r"(?P<dst_ip_port>0b[0-9a-f]+)\s+"
    r"(?P<src_port>\d+)\s+"
    r"(?P<dst_port>\d+)\s+"
    r"(?P<packet_type>\w)\s+"
)
U_tail = (
    r"(?P<sequence_number>\d+)\s+"
    r"(?P<tx_timestamp>\d+)\s+"
    r"(?P<priority_group>\d+)\s+"
    r"(?P<packet_size>\d+)\((?P<payload_size>\d+)\)"
)
A_tail = (
    r"0x(?P<ack_flags>[0-9a-fA-F]+) (?P<priority_group>\d+) (?P<sequence_number>\d+) (?P<tx_timestamp>\d+) (?P<packet_size>\d+)"
)
# Function to convert IP from binary representation to dotted notation
def convert_ip(bin_str):
    if isinstance(bin_str, str):
        return f"{int(bin_str[0:2], 16)}.{int(bin_str[2:4], 16)}.{int(bin_str[4:6], 16)}.{int(bin_str[6:], 16)}"
    return bin_str

# Function to parse a single line of the trace file
def parse_trace_line(line: str) -> dict:
    pattern_base_compile = re.compile(pattern_base)
    pattern_A = pattern_base + A_tail
    pattern_U = pattern_base + U_tail
    pattern_A_compile = re.compile(pattern_A)
    pattern_U_compile = re.compile(pattern_U)
    base = pattern_base_compile.match(line)
    if base == None:
        print("no base")
    packet_type = base['packet_type']
    if packet_type == 'A':
        result = pattern_A_compile.match(line)
        data = result.groupdict()
        #print(data)
        return data
    if packet_type == 'U':
        result = pattern_U_compile.match(line)
        data = result.groupdict()
        #print(data)
    else:
        print("other type")
    return data
# Function to create an iterator over the trace file to read and parse lines in batches
def trace_file_iterator(file_path: str, batch_size: int) -> Iterator[List[dict]]:
    with open(file_path, 'r') as file:
        while True:
            batch_data = []
            for _ in range(batch_size):
                line = file.readline()
                if not line:
                    break
                batch_data.append(parse_trace_line(line.strip()))
            
            # Fill the batch with None if it is smaller than batch_size
            while len(batch_data) < batch_size:
                batch_data.append(None)
            
            yield batch_data
            
            # Stop the iteration if we reached the end of the file
            if not line:
                break


if __name__ == '__main__':
    
    # Test the final implementation with the first batch of data
    file_path = 'trace8smaller.txt'
    batch_size = 200

    # Create the iterator
    trace_iterator = trace_file_iterator(file_path, batch_size)
    # Get the first batch of data
    while True:
        first_batch_data = next(trace_iterator)