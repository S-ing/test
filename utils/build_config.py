import yaml

def build_config(config_file='config/yolov12_level0.yaml'): # Default to Level0 config file
    with open(config_file, "r", encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    # Add config file path to the config for reference
    config['config_path'] = config_file

    if config['active_checker']:
        pass
    
    return config