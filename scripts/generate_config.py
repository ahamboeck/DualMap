import json
import random
import colorsys
import sys
import os

def generate_distinct_colors(n):
    """
    Generates N distinct colors using HSV space.
    Returns RGB values normalized between 0.0 and 1.0.
    """
    colors = []
    for i in range(n):
        # Golden ratio conjugate to spread hues evenly
        hue = (i * 0.618033988749895) % 1.0 
        saturation = 0.8 + (random.random() * 0.2) # High saturation (0.8-1.0)
        value = 0.8 + (random.random() * 0.2)      # High brightness (0.8-1.0)
        
        # Convert to RGB (result is already 0.0-1.0 floats)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(list(rgb))
    return colors

def main():
    # 1. Parse Command Line Arguments
    if len(sys.argv) < 2:
        print("Usage: python generate_config_arg.py <path_to_input_txt>")
        return

    input_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    # 2. Determine Output Paths
    # input: /data/my_list.txt 
    # outputs: /data/my_list_colors.json, /data/my_list_id_colors.json
    base_path_without_ext = os.path.splitext(input_path)[0]
    output_colors_json = f"{base_path_without_ext}_colors.json"
    output_id_colors_json = f"{base_path_without_ext}_id_colors.json"

    # 3. Read the input text file
    try:
        with open(input_path, 'r') as f:
            # Filter out empty lines and strip whitespace
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not classes:
        print("Error: The input file is empty.")
        return

    # 4. Generate Colors
    rgb_colors = generate_distinct_colors(len(classes))
    
    # 5. Prepare Data Structures
    
    # Structure A: "[name]_colors.json" -> {"class_name": [r, g, b]}
    colors_map = {}
    for i, class_name in enumerate(classes):
        colors_map[class_name] = rgb_colors[i]

    # Structure B: "[name]_id_colors.json" -> {"0": [r, g, b]}
    id_colors_map = {}
    for i in range(len(classes)):
        id_colors_map[str(i)] = rgb_colors[i]

    # 6. Save JSON files
    with open(output_colors_json, 'w') as f:
        json.dump(colors_map, f, indent=4)
        
    with open(output_id_colors_json, 'w') as f:
        json.dump(id_colors_map, f, indent=4)

    print(f"Successfully processed {len(classes)} classes.")
    print(f"Saved: {output_colors_json}")
    print(f"Saved: {output_id_colors_json}")

if __name__ == "__main__":
    main()