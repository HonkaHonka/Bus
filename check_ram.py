# check_ram.py
import psutil

def get_gb(bytes_value):
    """Converts bytes to gigabytes and rounds to two decimal places."""
    return round(bytes_value / (1024**3), 2)

def check_system_ram():
    """
    Checks and prints the system's total, available, and used RAM.
    """
    print("--- System Memory (RAM) Usage ---")
    
    # Get the memory details object
    virtual_mem = psutil.virtual_memory()
    
    # Extract total, available, and used RAM in bytes
    total_ram_bytes = virtual_mem.total
    available_ram_bytes = virtual_mem.available
    used_ram_bytes = virtual_mem.used
    
    # Get usage percentage
    usage_percent = virtual_mem.percent
    
    # Convert to Gigabytes for readability
    total_ram_gb = get_gb(total_ram_bytes)
    available_ram_gb = get_gb(available_ram_bytes)
    used_ram_gb = get_gb(used_ram_bytes)
    
    # Print the results in a clean format
    print(f"Total RAM:     {total_ram_gb} GB")
    print(f"Available RAM: {available_ram_gb} GB")
    print(f"Used RAM:      {used_ram_gb} GB")
    print(f"Usage Percent: {usage_percent}%")
    print("---------------------------------")


if __name__ == "__main__":
    check_system_ram()