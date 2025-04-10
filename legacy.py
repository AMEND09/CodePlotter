import re

def scale_gcode(gcode_lines, x_scale=1.0, y_scale=1.0, z_scale=1.0):
    """
    Scales the X, Y, and Z coordinates in a list of G-code lines.

    Args:
        gcode_lines (list): A list of strings, where each string is a line of G-code.
        x_scale (float): The scaling factor for the X coordinates.
        y_scale (float): The scaling factor for the Y coordinates.
        z_scale (float): The scaling factor for the Z coordinates.

    Returns:
        list: A new list of G-code lines with the scaling applied.
    """
    modified_gcode = []
    for line in gcode_lines:
        parts = line.split()
        modified_parts = []
        for part in parts:
            if part.startswith('X') and len(part) > 1:
                try:
                    x_val = float(part[1:]) * x_scale
                    modified_parts.append(f'X{x_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            elif part.startswith('Y') and len(part) > 1:
                try:
                    y_val = float(part[1:]) * y_scale
                    modified_parts.append(f'Y{y_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            elif part.startswith('Z') and len(part) > 1:
                try:
                    z_val = float(part[1:]) * z_scale
                    modified_parts.append(f'Z{z_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            else:
                modified_parts.append(part)
        modified_gcode.append(' '.join(modified_parts))
    return modified_gcode

def offset_gcode(gcode_lines, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    """
    Offsets the X, Y, and Z coordinates in a list of G-code lines.

    Args:
        gcode_lines (list): A list of strings, where each string is a line of G-code.
        x_offset (float): The amount to offset the X coordinates.
        y_offset (float): The amount to offset the Y coordinates.
        z_offset (float): The amount to offset the Z coordinates.

    Returns:
        list: A new list of G-code lines with the offsets applied.
    """
    modified_gcode = []
    for line in gcode_lines:
        parts = line.split()
        modified_parts = []
        for part in parts:
            if part.startswith('X') and len(part) > 1:
                try:
                    x_val = float(part[1:]) + x_offset
                    modified_parts.append(f'X{x_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            elif part.startswith('Y') and len(part) > 1:
                try:
                    y_val = float(part[1:]) + y_offset
                    modified_parts.append(f'Y{y_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            elif part.startswith('Z') and len(part) > 1:
                try:
                    z_val = float(part[1:]) + z_offset
                    modified_parts.append(f'Z{z_val:.6f}')
                except ValueError:
                    modified_parts.append(part)
            else:
                modified_parts.append(part)
        modified_gcode.append(' '.join(modified_parts))
    return modified_gcode

if __name__ == "__main__":
    try:
        file_path = input("Enter the path to your G-code file: ")
        with open(file_path, 'r') as f:
            gcode_lines = [line.strip() for line in f]

        print("\nOriginal G-code (first few lines):")
        for i, line in enumerate(gcode_lines[:5]):
            print(f"{i+1}: {line}")
        if len(gcode_lines) > 5:
            print("...")
            print(f"Total lines: {len(gcode_lines)}")

        x_scale = float(input("Enter the X-axis scale factor (e.g., 2.0 for 200%): "))
        y_scale = float(input("Enter the Y-axis scale factor: "))
        z_scale = float(input("Enter the Z-axis scale factor: "))

        scaled_code = scale_gcode(gcode_lines, x_scale, y_scale, z_scale)

        x_offset = float(input("Enter the X-axis offset: "))
        y_offset = float(input("Enter the Y-axis offset: "))
        z_offset = float(input("Enter the Z-axis offset: "))

        modified_code = offset_gcode(scaled_code, x_offset, y_offset, z_offset)

        output_file_path = input("Enter the path to save the modified G-code file: ")
        with open(output_file_path, 'w') as outfile:
            for line in modified_code:
                outfile.write(line + '\n')

        print(f"\nModified G-code (scaled and offset) saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except ValueError:
        print("Invalid input. Please enter numeric values for the scaling factors and offsets.")
    except Exception as e:
        print(f"An error occurred: {e}")