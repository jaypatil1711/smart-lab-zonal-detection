def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def load_config(config_file):
    """Loads configuration from a given file."""
    import json
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_config(config_file, config_data):
    """Saves configuration to a given file."""
    import json
    with open(config_file, 'w') as file:
        json.dump(config_data, file, indent=4)

def calculate_fps(start_time, end_time, frame_count):
    """Calculates frames per second (FPS)."""
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps