"""
GCode Visualizer & Manipulator - A tool for visualizing, transforming, and merging G-code files

This application provides a graphical interface for:
- Loading and viewing multiple G-code files
- Transforming G-code with scale, rotation, and translation
- Warping bed images to match real-world coordinates
- Merging multiple G-code files with individual transformations
- Visualizing G-code paths with accurate bed representation
- Supporting pen plotters with pen offset adjustments

Created by Aditya Mendiratta
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import re
import math
import os
import itertools # For color cycling
import sys # For checking platform for scroll wheel binding

# --- Image Processing Dependencies ---
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    IMAGE_PROCESSING_ENABLED = True
except ImportError:
    IMAGE_PROCESSING_ENABLED = False
    print("WARNING: OpenCV (cv2), NumPy, or Pillow not found. Image warping features disabled.")
    print("Install with: pip install opencv-python numpy Pillow")

# --- Constants ---
GCODE_TOLERANCE = 1e-6 # For comparing float coordinates
PAPER_SIZES = {
    "None": None,
    "A4": (210.0, 297.0),         # mm
    "US Letter": (215.9, 279.4),  # mm (8.5 x 11 inches)
    "College Ruled": (203.2, 266.7) # mm (Approx 8 x 10.5 inches usable)
}
# Define a cycle of colors for loaded files
DEFAULT_COLORS = itertools.cycle(['blue', 'green', 'red', 'purple', 'darkorange', 'cyan', 'magenta', 'brown', 'olive'])
CORNER_NAMES = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]


# --- Core G-code processing functions ---
# (parse_gcode_for_xy, apply_transformations_to_coords, generate_transformed_gcode remain unchanged)
def parse_gcode_for_xy(gcode_lines):
    """
    Extracts X, Y coordinates and command type (G0/G1) for drawing.
    Each file is parsed independently, starting implicitly from (0,0)
    unless G92 changes it within the file.
    Returns segments and the final X,Y position *relative to the file's start*.
    """
    path_segments = []
    current_segment = []
    last_x, last_y = 0.0, 0.0 # Assume start at 0,0 for this file's relative coords
    in_segment = False

    for line_num, line in enumerate(gcode_lines):
        original_line = line
        line = line.strip().upper().split(';')[0]
        if not line: continue

        parts = line.split()
        command = None
        x_coord, y_coord = None, None # Store explicit coords found on this line
        z_present = False
        xy_present = False
        is_g0_g1 = False
        is_g92 = False

        # --- Parse command and coordinates ---
        temp_x, temp_y = last_x, last_y # Start with last known position

        for part in parts:
            if not part: continue
            char = part[0]
            try:
                val_str = part[1:]
                if char == 'G':
                    g_val = int(float(val_str))
                    if g_val == 0 or g_val == 1:
                        command = f"G{g_val}"
                        is_g0_g1 = True
                    elif g_val == 92: # Set Position
                        is_g92 = True
                elif char == 'X':
                    x_coord = float(val_str)
                    temp_x = x_coord # Directly update temp pos
                    xy_present = True
                elif char == 'Y':
                    y_coord = float(val_str)
                    temp_y = y_coord # Directly update temp pos
                    xy_present = True
                elif char == 'Z':
                    z_present = True
            except (ValueError, IndexError):
                continue

        # --- Logic for points/segments ---
        if is_g0_g1:
            is_z_only_move = z_present and not xy_present
            # If X or Y was not specified, use the last known value
            current_point_x = x_coord if x_coord is not None else last_x
            current_point_y = y_coord if y_coord is not None else last_y

            current_point = {'x': current_point_x, 'y': current_point_y, 'cmd': command, 'z_only': is_z_only_move, 'line': line_num}
            moved_xy = abs(last_x - current_point_x) > GCODE_TOLERANCE or abs(last_y - current_point_y) > GCODE_TOLERANCE

            if command == "G0":
                if current_segment:
                    path_segments.append(list(current_segment))
                if moved_xy:
                    start_point_g0 = {'x': last_x, 'y': last_y, 'cmd': "G0_START", 'z_only': False, 'line': line_num}
                    g0_segment = [start_point_g0, current_point]
                    path_segments.append(g0_segment)
                current_segment = []
                in_segment = False

            elif command == "G1":
                if not in_segment and moved_xy:
                     start_point_g1 = {'x': last_x, 'y': last_y, 'cmd': "G1", 'z_only': False, 'line': line_num}
                     current_segment.append(start_point_g1)

                if moved_xy:
                     current_segment.append(current_point)
                     in_segment = True
                elif is_z_only_move and in_segment:
                     path_segments.append(list(current_segment))
                     current_segment = []
                     in_segment = False
                elif not moved_xy and not current_segment: # First G1 is Z-only
                     start_point_g1 = {'x': last_x, 'y': last_y, 'cmd': "G1", 'z_only': False, 'line': line_num}
                     current_segment.append(start_point_g1)
                     current_segment.append(current_point) # Add the Z-only point
                     in_segment = True

            # Update last position
            last_x, last_y = current_point_x, current_point_y

        elif is_g92: # Handle G92: Set new logical position
             if in_segment: # End current segment before position reset
                 path_segments.append(list(current_segment))
                 current_segment = []
                 in_segment = False
             # Update last_x/last_y if X/Y specified in G92
             if x_coord is not None: last_x = x_coord
             if y_coord is not None: last_y = y_coord
             # Don't add a visual segment for G92 itself

        else: # Other commands
            if in_segment: # End current segment
                 path_segments.append(list(current_segment))
                 current_segment = []
                 in_segment = False
            # Update position if X/Y changed implicitly by other commands (rare for plotting)
            # We already updated temp_x/y if X/Y were present, commit them if command wasn't handled above
            if xy_present: # If some other command included X/Y
                last_x, last_y = temp_x, temp_y


    if current_segment:
        path_segments.append(current_segment)

    # Return segments and the final relative X, Y
    return [seg for seg in path_segments if len(seg) >= 2], last_x, last_y

# --- Coordinate Transformation ---
def apply_transformations_to_coords(original_coords_segments, transform_params):
    """Applies scale, offset, and pen offset from a transform dict to coordinate segments."""
    transformed_segments = []
    x_scale = transform_params['x_scale']
    y_scale = transform_params['y_scale']
    x_offset = transform_params['x_offset']
    y_offset = transform_params['y_offset']
    
    # Get rotation angle in radians if it exists, otherwise default to 0
    rotation_angle = math.radians(transform_params.get('rotation_angle', 0))
    
    # Do NOT apply pen offsets for preview display
    # pen_x_offset = transform_params['pen_x_offset']
    # pen_y_offset = transform_params['pen_y_offset']

    for segment in original_coords_segments:
        new_segment = []
        for point in segment:
            # Apply scale to original relative coords
            scaled_x = point['x'] * x_scale
            scaled_y = point['y'] * y_scale
            
            # Apply rotation if needed (rotate around origin before translation)
            if rotation_angle != 0:
                # Rotation formula: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
                cos_theta = math.cos(rotation_angle)
                sin_theta = math.sin(rotation_angle)
                rotated_x = scaled_x * cos_theta - scaled_y * sin_theta
                rotated_y = scaled_x * sin_theta + scaled_y * cos_theta
                transformed_x = rotated_x + x_offset
                transformed_y = rotated_y + y_offset
            else:
                # No rotation, just apply offset
                transformed_x = scaled_x + x_offset
                transformed_y = scaled_y + y_offset
            
            # Copy other info
            new_segment.append({**point, 'x': transformed_x, 'y': transformed_y})
        transformed_segments.append(new_segment)
    return transformed_segments

# --- G-code Generation ---
def generate_transformed_gcode(original_gcode_lines, transform_params):
    """Generates modified G-code lines applying transformations from a transform dict."""
    modified_gcode = []
    coord_pattern = re.compile(r"([XYZ])([-+]?\d*\.?\d*)")

    x_scale = transform_params['x_scale']
    y_scale = transform_params['y_scale']
    z_scale = transform_params['z_scale']
    x_offset = transform_params['x_offset']
    y_offset = transform_params['y_offset']
    z_offset = transform_params['z_offset']
    pen_x_offset = transform_params['pen_x_offset']
    pen_y_offset = transform_params['pen_y_offset']
    
    # Get rotation angle in radians if it exists, otherwise default to 0
    rotation_angle = math.radians(transform_params.get('rotation_angle', 0))
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)

    for line in original_gcode_lines:
        modified_line = line
        parts = line.split(';', 1)
        code_part = parts[0]
        comment_part = f";{parts[1]}" if len(parts) > 1 else ""

        cleaned_code_upper = code_part.strip().upper()
        is_motion_cmd = cleaned_code_upper.startswith('G0') or cleaned_code_upper.startswith('G1')
        is_set_coord_cmd = cleaned_code_upper.startswith('G92')

        # For commands with coordinates, we need to extract all coordinates first,
        # then apply transformations to all of them, then rebuild the line
        if is_motion_cmd or is_set_coord_cmd:
            # Extract all coordinates from this line
            matches = list(coord_pattern.finditer(code_part))
            if matches:
                axes_values = {}  # Store original values by axis
                for match in matches:
                    axis = match.group(1).upper()
                    try:
                        val = float(match.group(2))
                        axes_values[axis] = val
                    except ValueError:
                        pass
                
                # Apply transformations to X and Y coordinates if both are present
                if 'X' in axes_values and 'Y' in axes_values:
                    # Apply scaling
                    scaled_x = axes_values['X'] * x_scale
                    scaled_y = axes_values['Y'] * y_scale
                    
                    # Apply rotation
                    if rotation_angle != 0:
                        rotated_x = scaled_x * cos_theta - scaled_y * sin_theta
                        rotated_y = scaled_x * sin_theta + scaled_y * cos_theta
                        scaled_x = rotated_x
                        scaled_y = rotated_y
                    
                    # Apply offsets
                    final_x = scaled_x + x_offset - pen_x_offset  # Subtract pen offset
                    final_y = scaled_y + y_offset - pen_y_offset  # Subtract pen offset
                    
                    # Update values
                    axes_values['X'] = final_x
                    axes_values['Y'] = final_y
                
                # Handle Z separately (no rotation)
                if 'Z' in axes_values:
                    axes_values['Z'] = (axes_values['Z'] * z_scale) + z_offset
                
                # Rebuild the line with transformed coordinates
                modified_code_part = cleaned_code_upper.split()[0]  # Get command (G0, G1, G92)
                for axis, value in sorted(axes_values.items()):  # Sort by axis for consistency
                    modified_code_part += f" {axis}{value:.6f}"
                
                modified_line = modified_code_part + comment_part
            else:
                modified_line = code_part + comment_part
        else:
            # For non-motion commands, keep as is
            modified_line = code_part + comment_part
        
        modified_gcode.append(modified_line)

    return modified_gcode


# --- GUI Application Class ---

class GCodeVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("G-Code Visualizer & Manipulator (Multi-File)")
        self.geometry("1250x850") # Wider for file list + new controls

        # --- Recursion Prevention ---
        self._updating_bed_params = False

        # --- Data Storage ---
        self.gcode_files = [] # List of dictionaries, one per file
        self.selected_file_index = tk.IntVar(value=-1) # Track listbox selection index

        # --- Bed Image Data ---
        self.original_bed_image_pil = None # Stores the loaded PIL Image
        # self.original_bed_image_tk = None # No longer needed globally, used temporarily
        self.original_bed_image_cv = None # Stores the OpenCV image (numpy array)
        self.warped_bed_image_pil = None  # Stores the high-res warped PIL image (canonical)
        self.warped_bed_image_tk = None   # Stores the currently displayed Tkinter PhotoImage (dynamically resized)
        self.bed_image_corners_src = []   # Stores [(x,y), ...] in ORIGINAL image coords for bed
        self.paper_image_corners_src = [] # Stores [(x,y), ...] in ORIGINAL image coords for paper
        self.corner_selection_mode = None # None, 'bed', 'paper'
        self.corner_selection_points_temp = [] # Temporary storage during clicking
        self.corner_selection_markers = [] # Canvas IDs for click markers
        self.temp_image_display_id = None # Canvas ID for the temporary original image display
        self.image_display_scale_factor = 1.0 # Scale factor used to display temp image
        self.PIXELS_PER_MM = 5 # Resolution for the canonical warped image

        # --- Transformation State Variables (reflect selected file) ---
        self.current_x_scale = tk.DoubleVar(value=1.0)
        self.current_y_scale = tk.DoubleVar(value=1.0)
        self.current_z_scale = tk.DoubleVar(value=1.0)
        self.current_x_offset = tk.DoubleVar(value=0.0)
        self.current_y_offset = tk.DoubleVar(value=0.0)
        self.current_z_offset = tk.DoubleVar(value=0.0)
        self.pen_offset_x_var = tk.DoubleVar(value=0.0)
        self.pen_offset_y_var = tk.DoubleVar(value=0.0)
        self.current_visibility_var = tk.BooleanVar(value=True)

        # --- Bed Settings State ---
        self.bed_width_var = tk.DoubleVar(value=220.0)
        self.bed_height_var = tk.DoubleVar(value=220.0)

        # --- Paper Overlay State ---
        self.paper_type_var = tk.StringVar(value="None")
        self.paper_offset_x_var = tk.DoubleVar(value=0.0)
        self.paper_offset_y_var = tk.DoubleVar(value=0.0)

        # --- View State ---
        self.hide_z_moves_var = tk.BooleanVar(value=False)
        self.show_warped_image_var = tk.BooleanVar(value=False) # For toggling background
        self.skip_pen_warning_var = tk.BooleanVar(value=False) # For skipping pen offset out-of-bounds warning

        # --- Canvas Setup ---
        # Adjusted for potential wider controls frame
        self.canvas_view_width = 750
        self.canvas_view_height = 750
        self.padding = 30

        # --- UI Elements ---
        self.create_widgets()

        # --- Mouse Bindings ---
        # Existing bindings
        self.canvas.tag_bind("gcode_path", "<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop) # General release needed
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start) # Middle mouse
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_stop)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel) # Zoom (basic)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel) # Zoom Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel) # Zoom Linux
        # Binding for Alt+Click pan (if middle mouse is tricky)
        self.canvas.bind("<Alt-ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<Alt-ButtonRelease-1>", self.on_pan_stop)
        self.canvas.bind("<Alt-B1-Motion>", self.on_pan_motion)

        # New binding specifically for corner selection (will be managed dynamically)
        self._corner_click_binding_id = None

        # --- Drag/Pan State Variables ---
        self._pan_data = {"x": 0, "y": 0}
        self._drag_data = {
            "start_offset_x": 0, "start_offset_y": 0,
            "last_event_x": 0, "last_event_y": 0,
            "is_dragging": False,
            "dragged_file_index": -1
        }

        # --- Initialize ---
        self.update_bed_display_params()
        self.draw_bed()
        self.update_ui_for_selection() # Disable controls initially
        self.update_image_controls_state() # Disable image controls initially


    # --- Coordinate Conversion (Adapt gcode_to_canvas slightly) ---
    def update_bed_display_params(self):
        """Recalculates scaling and corner pixels based on bed size vars and canvas view size."""
        # Prevent recursion
        if hasattr(self, '_updating_bed_params') and self._updating_bed_params:
            return False
        self._updating_bed_params = True

        success = False
        try:
            # Get values robustly
            try:
                bed_width_mm = self.bed_width_var.get()
            except (tk.TclError, ValueError):
                self.bed_width_var.set(220.0)
                raise ValueError("Invalid width value entered.")

            try:
                bed_height_mm = self.bed_height_var.get()
            except (tk.TclError, ValueError):
                self.bed_height_var.set(220.0)
                raise ValueError("Invalid height value entered.")

            if bed_width_mm <= 0 or bed_height_mm <= 0:
                raise ValueError("Bed dimensions must be positive.")

            # Store valid values and perform calculations
            self.bed_width_mm = bed_width_mm
            self.bed_height_mm = bed_height_mm

            available_width = self.canvas_view_width - 2 * self.padding
            available_height = self.canvas_view_height - 2 * self.padding
            if available_width <= 0 or available_height <= 0:
                available_width = available_height = 1

            scale_x = available_width / self.bed_width_mm if self.bed_width_mm > 0 else float('inf')
            scale_y = available_height / self.bed_height_mm if self.bed_height_mm > 0 else float('inf')
            self.canvas_scale = min(scale_x, scale_y)
            if self.canvas_scale == float('inf'):
                self.canvas_scale = 1.0

            bed_pixel_width = self.bed_width_mm * self.canvas_scale
            bed_pixel_height = self.bed_height_mm * self.canvas_scale

            self.bed_cx_min = self.padding + (available_width - bed_pixel_width) / 2
            self.bed_cy_max = self.canvas_view_height - self.padding - (available_height - bed_pixel_height) / 2
            self.bed_cx_max = self.bed_cx_min + bed_pixel_width
            self.bed_cy_min = self.bed_cy_max - bed_pixel_height

            # Recalculate warp if image corners are set
            if IMAGE_PROCESSING_ENABLED and self.original_bed_image_pil and len(self.bed_image_corners_src) == 4:
                self.calculate_and_set_warp()

            success = True

        except ValueError as e:
            messagebox.showerror("Bed Size Error", f"Invalid bed dimensions: {e}")
            self.bed_width_var.set(220.0)
            self.bed_height_var.set(220.0)
            self._calculate_default_bed_params()
            success = False

        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred updating bed parameters:\n{e}")
            self.bed_width_var.set(220.0)
            self.bed_height_var.set(220.0)
            self._calculate_default_bed_params()
            success = False

        finally:
            self._updating_bed_params = False

        return success

    def _calculate_default_bed_params(self):
        """Helper to calculate bed parameters using default 220x220 dimensions."""
        self.bed_width_mm = 220.0
        self.bed_height_mm = 220.0
        available_width = self.canvas_view_width - 2 * self.padding
        available_height = self.canvas_view_height - 2 * self.padding
        if available_width <= 0 or available_height <= 0:
            available_width = available_height = 1

        scale_x = available_width / self.bed_width_mm
        scale_y = available_height / self.bed_height_mm
        self.canvas_scale = min(scale_x, scale_y)
        if self.canvas_scale == float('inf'):
            self.canvas_scale = 1.0

        bed_pixel_width = self.bed_width_mm * self.canvas_scale
        bed_pixel_height = self.bed_height_mm * self.canvas_scale
        self.bed_cx_min = self.padding + (available_width - bed_pixel_width) / 2
        self.bed_cy_max = self.canvas_view_height - self.padding - (available_height - bed_pixel_height) / 2
        self.bed_cx_max = self.bed_cx_min + bed_pixel_width
        self.bed_cy_min = self.bed_cy_max - bed_pixel_height


    def update_bed_action(self):
        """Action for the Update Bed button."""
        if self.update_bed_display_params():
            # Redrawing bed, warp, and gcode is necessary
            self.redraw_canvas()
            self.status_label.config(text="Bed dimensions updated.")

    def gcode_to_canvas(self, x, y):
        """Converts G-code coordinates (mm, origin bottom-left) to canvas pixel coordinates."""
        # Ensure bed params are valid before calculating
        if not hasattr(self, 'bed_cx_min'):
             if not self.update_bed_display_params():
                 return 0, 0 # Return default if bed update fails

        cx = self.bed_cx_min + (x * self.canvas_scale)
        cy = self.bed_cy_max - (y * self.canvas_scale) # Flip Y axis
        return cx, cy

    def canvas_to_gcode(self, cx, cy):
        """Converts canvas pixel coordinates back to G-code coordinates (mm)."""
        # Ensure bed params are valid
        if not hasattr(self, 'bed_cx_min') or not hasattr(self, 'canvas_scale') or abs(self.canvas_scale) < 1e-9:
            if not self.update_bed_display_params() or abs(self.canvas_scale) < 1e-9:
                return 0.0, 0.0 # Return default if bed update fails

        # Account for canvas scrolling (convert view coords to canvas coords)
        canvas_x_in_scroll_region = self.canvas.canvasx(cx)
        canvas_y_in_scroll_region = self.canvas.canvasy(cy)

        gx = (canvas_x_in_scroll_region - self.bed_cx_min) / self.canvas_scale
        gy = (self.bed_cy_max - canvas_y_in_scroll_region) / self.canvas_scale # Flip Y axis back
        return gx, gy


    def create_widgets(self):
        # Main Frames
        left_frame = ttk.Frame(self, width=250) # Fixed width for file list etc.
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # Create a canvas and scrollbar for the controls frame to make it scrollable
        self.controls_canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        
        # Pack the canvas and scrollbar
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        controls_scrollbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5), pady=5)
        
        # Create the inner frame for controls that will be placed in the canvas
        controls_frame = ttk.Frame(self.controls_canvas)
        self.controls_frame = controls_frame
        
        # Create a window in the canvas to hold the controls frame
        self.controls_canvas_window = self.controls_canvas.create_window(
            (0, 0), window=controls_frame, anchor="nw", tags="controls_frame"
        )
        
        # Configure canvas size and scrolling behavior
        controls_frame.bind("<Configure>", self.on_controls_frame_configure)
        self.controls_canvas.bind("<Configure>", self.on_controls_canvas_configure)
        self.controls_canvas.bind_all("<MouseWheel>", self.on_mousewheel_controls)
        self.controls_canvas.bind_all("<Button-4>", self.on_mousewheel_controls)
        self.controls_canvas.bind_all("<Button-5>", self.on_mousewheel_controls)

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left Frame Widgets (File List & Management) ---
        # (Unchanged from previous version)
        file_list_frame = ttk.LabelFrame(left_frame, text="Loaded Files", padding=5)
        file_list_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.file_listbox = tk.Listbox(file_list_frame, exportselection=False)
        self.file_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        file_buttons_frame = ttk.Frame(file_list_frame)
        file_buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
        self.load_button = ttk.Button(file_buttons_frame, text="Load GCode", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.remove_button = ttk.Button(file_buttons_frame, text="Remove", command=self.remove_selected_file, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.clear_all_button = ttk.Button(file_buttons_frame, text="Clear All", command=self.reset_app_state, state=tk.DISABLED)
        self.clear_all_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)


        # --- Controls Frame Widgets ---
        controls_frame.columnconfigure(0, weight=1) # Make column expandable
        row_idx = 0

        # --- File Operations ---
        file_op_frame = ttk.LabelFrame(controls_frame, text="File Operations", padding=5)
        file_op_frame.grid(row=row_idx, column=0, pady=(0,5), sticky="ew"); row_idx += 1
        self.save_button = ttk.Button(file_op_frame, text="Save Visible Merged GCode", command=self.save_file, state=tk.DISABLED)
        self.save_button.pack(pady=2, fill=tk.X)

        # --- Selected File Controls ---
        selected_file_frame = ttk.LabelFrame(controls_frame, text="Selected File Controls", padding=5)
        selected_file_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        self.visible_check = ttk.Checkbutton(selected_file_frame, text="Visible", variable=self.current_visibility_var, command=self.toggle_visibility, state=tk.DISABLED)
        self.visible_check.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.color_button = ttk.Button(selected_file_frame, text="Color", command=self.change_color, state=tk.DISABLED)
        self.color_button.grid(row=0, column=1, padx=5, pady=2)
        self.reset_button = ttk.Button(selected_file_frame, text="Reset Transforms", command=self.reset_selected_file_transforms, state=tk.DISABLED)
        self.reset_button.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        selected_file_frame.columnconfigure(2, weight=1)


        # --- Bed Settings ---
        bed_frame = ttk.LabelFrame(controls_frame, text="Bed Settings (mm)", padding="5")
        bed_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        ttk.Label(bed_frame, text="Width:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.bed_width_entry = ttk.Entry(bed_frame, textvariable=self.bed_width_var, width=7)
        self.bed_width_entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(bed_frame, text="Height:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.bed_height_entry = ttk.Entry(bed_frame, textvariable=self.bed_height_var, width=7)
        self.bed_height_entry.grid(row=1, column=1, padx=2, pady=2)
        self.update_bed_button = ttk.Button(bed_frame, text="Update Bed", command=self.update_bed_action)
        self.update_bed_button.grid(row=0, column=2, rowspan=2, padx=10, pady=2, sticky="ns")


        # --- Paper Overlay ---
        paper_frame = ttk.LabelFrame(controls_frame, text="Virtual Paper Overlay", padding="5")
        paper_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        ttk.Label(paper_frame, text="Type:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.paper_combo = ttk.Combobox(paper_frame, textvariable=self.paper_type_var, values=list(PAPER_SIZES.keys()), width=12, state="readonly")
        self.paper_combo.grid(row=0, column=1, columnspan=2, padx=2, pady=2, sticky="ew")
        self.paper_combo.bind("<<ComboboxSelected>>", self.on_paper_type_change)
        ttk.Label(paper_frame, text="Offset X:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.paper_offset_x_entry = ttk.Entry(paper_frame, textvariable=self.paper_offset_x_var, width=7)
        self.paper_offset_x_entry.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(paper_frame, text="Y:").grid(row=1, column=2, padx=2, pady=2, sticky="w")
        self.paper_offset_y_entry = ttk.Entry(paper_frame, textvariable=self.paper_offset_y_var, width=7)
        self.paper_offset_y_entry.grid(row=1, column=3, padx=2, pady=2)
        self.update_paper_button = ttk.Button(paper_frame, text="Update Overlay", command=self.update_paper_overlay_action)
        self.update_paper_button.grid(row=0, column=3, rowspan=1, padx=5, pady=2, sticky="e")

        # --- NEW: Bed Image Alignment ---
        image_frame = ttk.LabelFrame(controls_frame, text="Bed Image Alignment", padding="5")
        image_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        image_frame.columnconfigure(0, weight=1) # Allow buttons/labels to expand

        self.load_image_button = ttk.Button(image_frame, text="Load Bed Image", command=self.load_bed_image)
        self.load_image_button.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="ew")

        self.select_bed_corners_button = ttk.Button(image_frame, text="Select Bed Corners (4)", command=lambda: self.start_corner_selection('bed'), state=tk.DISABLED)
        self.select_bed_corners_button.grid(row=1, column=0, padx=2, pady=2, sticky="ew")

        self.select_paper_corners_button = ttk.Button(image_frame, text="Select Paper Corners (4)", command=lambda: self.start_corner_selection('paper'), state=tk.DISABLED)
        self.select_paper_corners_button.grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        self.corner_status_label = ttk.Label(image_frame, text="Load image first", wraplength=200, justify=tk.LEFT)
        self.corner_status_label.grid(row=2, column=0, columnspan=2, padx=2, pady=3, sticky="w")

        self.show_warped_image_check = ttk.Checkbutton(image_frame, text="Show Warped Image", variable=self.show_warped_image_var, command=self.toggle_warped_image_display, state=tk.DISABLED)
        self.show_warped_image_check.grid(row=3, column=0, padx=2, pady=2, sticky="w")

        self.clear_image_button = ttk.Button(image_frame, text="Clear Image Data", command=self.clear_image_data, state=tk.DISABLED)
        self.clear_image_button.grid(row=3, column=1, padx=2, pady=2, sticky="e")

        # Disable image features if dependencies are missing
        if not IMAGE_PROCESSING_ENABLED:
             self.load_image_button.config(state=tk.DISABLED)
             self.corner_status_label.config(text="Error: OpenCV/NumPy/Pillow missing.")
             for widget in image_frame.winfo_children():
                 if widget != self.corner_status_label: # Keep the error message visible
                     widget.config(state=tk.DISABLED)

        # --- Global View Options ---
        view_frame = ttk.LabelFrame(controls_frame, text="Global View Options", padding="5")
        view_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        self.hide_z_check = ttk.Checkbutton(view_frame, text="Hide Z-only moves (All Files)", variable=self.hide_z_moves_var, command=self.redraw_canvas)
        self.hide_z_check.pack(anchor="w")
        
        self.skip_pen_warning_check = ttk.Checkbutton(view_frame, text="Skip pen offset warning when saving", variable=self.skip_pen_warning_var)
        self.skip_pen_warning_check.pack(anchor="w", pady=(5,0))

        # --- Transformation Entries ---
        transform_frame = ttk.LabelFrame(controls_frame, text="Transform Selected File", padding=5)
        transform_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        transform_frame.columnconfigure(1, weight=1)

        # NEW: Rotation entry
        ttk.Label(transform_frame, text="Rotate:").grid(row=0, column=0, padx=2, pady=1, sticky="w")
        self.rotation_angle_var = tk.DoubleVar(value=0.0)
        self.rotation_entry = ttk.Entry(transform_frame, textvariable=self.rotation_angle_var, width=10, state=tk.DISABLED)
        self.rotation_entry.grid(row=0, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="°").grid(row=0, column=2, padx=0, pady=1, sticky="w")

        # Scale entries
        ttk.Label(transform_frame, text="Scale X:").grid(row=1, column=0, padx=2, pady=1, sticky="w")
        self.scale_x_entry = ttk.Entry(transform_frame, textvariable=self.current_x_scale, width=10, state=tk.DISABLED)
        self.scale_x_entry.grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        ttk.Label(transform_frame, text="Scale Y:").grid(row=2, column=0, padx=2, pady=1, sticky="w")
        self.scale_y_entry = ttk.Entry(transform_frame, textvariable=self.current_y_scale, width=10, state=tk.DISABLED)
        self.scale_y_entry.grid(row=2, column=1, padx=2, pady=1, sticky="ew")

        ttk.Label(transform_frame, text="Scale Z:").grid(row=3, column=0, padx=2, pady=1, sticky="w")
        self.scale_z_entry = ttk.Entry(transform_frame, textvariable=self.current_z_scale, width=10, state=tk.DISABLED)
        self.scale_z_entry.grid(row=3, column=1, padx=2, pady=1, sticky="ew")

        ttk.Label(transform_frame, text="Offset X:").grid(row=4, column=0, padx=2, pady=1, sticky="w")
        self.offset_x_entry = ttk.Entry(transform_frame, textvariable=self.current_x_offset, width=10, state=tk.DISABLED)
        self.offset_x_entry.grid(row=4, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Offset Y:").grid(row=5, column=0, padx=2, pady=1, sticky="w")
        self.offset_y_entry = ttk.Entry(transform_frame, textvariable=self.current_y_offset, width=10, state=tk.DISABLED)
        self.offset_y_entry.grid(row=5, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Offset Z:").grid(row=6, column=0, padx=2, pady=1, sticky="w")
        self.offset_z_entry = ttk.Entry(transform_frame, textvariable=self.current_z_offset, width=10, state=tk.DISABLED)
        self.offset_z_entry.grid(row=6, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Pen Offset X:").grid(row=7, column=0, padx=2, pady=1, sticky="w")
        self.pen_offset_x_entry = ttk.Entry(transform_frame, textvariable=self.pen_offset_x_var, width=10, state=tk.DISABLED)
        self.pen_offset_x_entry.grid(row=7, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Pen Offset Y:").grid(row=8, column=0, padx=2, pady=1, sticky="w")
        self.pen_offset_y_entry = ttk.Entry(transform_frame, textvariable=self.pen_offset_y_var, width=10, state=tk.DISABLED)
        self.pen_offset_y_entry.grid(row=8, column=1, padx=2, pady=1, sticky="ew")
        self.apply_transforms_button = ttk.Button(transform_frame, text="Apply to Selected", command=self.apply_transformations_to_selected, state=tk.DISABLED)
        self.apply_transforms_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # --- Actions on Selected File ---
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions on Selected File", padding="5")
        actions_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        self.center_button = ttk.Button(actions_frame, text="Center Selected on Bed", command=self.center_selected_on_bed, state=tk.DISABLED)
        self.center_button.pack(pady=2, fill=tk.X)
        self.ender3_offset_button = ttk.Button(actions_frame, text="Set Selected Margin (10,10)", command=self.apply_ender3_offset_to_selected, state=tk.DISABLED)
        self.ender3_offset_button.pack(pady=2, fill=tk.X)

        # --- Status Display ---
        status_frame = ttk.Frame(controls_frame, padding="5")
        status_frame.grid(row=row_idx, column=0, pady=(10,0), sticky="sew")
        controls_frame.rowconfigure(row_idx, weight=1) # Push status to bottom
        self.files_loaded_label = ttk.Label(status_frame, text="Files loaded: 0", wraplength=280, justify=tk.LEFT)
        self.files_loaded_label.pack(pady=1, anchor='w')
        self.bounds_label = ttk.Label(status_frame, text="Visible Bounds: N/A", wraplength=280, justify=tk.LEFT)
        self.bounds_label.pack(pady=1, anchor='w')
        self.status_label = ttk.Label(status_frame, text="Status: Ready.", wraplength=280, justify=tk.LEFT)
        self.status_label.pack(pady=1, anchor='w')


        # --- Canvas & Scrollbars ---
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_view_width, height=self.canvas_view_height, bg="ivory", relief=tk.SUNKEN, borderwidth=1)
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)


    # --- File Management & Selection ---
    # (load_file, on_file_select, update_ui_for_selection, clear_transform_entries,
    #  remove_selected_file, reset_app_state, update_status_files_loaded remain unchanged)
    def load_file(self):
        filepaths = filedialog.askopenfilenames(
            title="Open G-Code File(s)",
            filetypes=[("G-Code Files", "*.gcode *.nc *.g"), ("All Files", "*.*")]
        )
        if not filepaths: return

        # Remove any temporary original image when loading new gcode files
        if self.temp_image_display_id:
            self.canvas.delete("temp_image_display")
            self.temp_image_display_id = None
            self.temp_original_bed_image_tk = None

        newly_added_indices = []
        for filepath in filepaths:
            # Avoid adding duplicates based on path
            if any(f['filepath'] == filepath for f in self.gcode_files):
                self.status_label.config(text=f"File already loaded: {os.path.basename(filepath)}")
                continue
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    self.status_label.config(text=f"Skipped empty file: {os.path.basename(filepath)}")
                    continue

                # Parse independently
                original_segments, _, _ = parse_gcode_for_xy(lines)

                # Create new file data entry
                file_data = {
                    'filepath': filepath,
                    'basename': os.path.basename(filepath),
                    'original_lines': lines,
                    'original_segments': original_segments,
                    'transform': { # Default transforms for this file
                        'x_scale': 1.0, 'y_scale': 1.0, 'z_scale': 1.0,
                        'x_offset': 0.0, 'y_offset': 0.0, 'z_offset': 0.0,
                        'pen_x_offset': 0.0, 'pen_y_offset': 0.0
                    },
                    'current_segments': [], # Will be calculated on demand or apply
                    'color': next(DEFAULT_COLORS),
                    'visible': True,
                    'id': len(self.gcode_files) # Use index as simple ID
                }
                # Calculate initial current_segments
                file_data['current_segments'] = apply_transformations_to_coords(
                    file_data['original_segments'], file_data['transform']
                )

                self.gcode_files.append(file_data)
                self.file_listbox.insert(tk.END, file_data['basename'])
                newly_added_indices.append(len(self.gcode_files) - 1)

            except Exception as e:
                messagebox.showerror("Error Loading File", f"Could not load or parse file:\n{filepath}\n\n{e}")
                self.status_label.config(text="Error loading file.")

        if newly_added_indices:
            # Select the first newly added file
            last_added_index = newly_added_indices[-1]
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(last_added_index)
            self.file_listbox.see(last_added_index)
            self.selected_file_index.set(last_added_index)
            self.on_file_select() # Trigger UI update for selection
            self.redraw_canvas()
            self.update_status_files_loaded()
            self.status_label.config(text="File(s) loaded.")
            # Enable global buttons if needed
            self.clear_all_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)


    def on_file_select(self, event=None):
        """Handles selection change in the file listbox."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            self.selected_file_index.set(-1)
            self.clear_transform_entries() # Clear display if nothing selected
        else:
            index = selected_indices[0]
            self.selected_file_index.set(index)
            if 0 <= index < len(self.gcode_files):
                file_data = self.gcode_files[index]
                transform = file_data['transform']
                # Update the DoubleVars to reflect the selected file's transforms
                self.current_x_scale.set(transform['x_scale'])
                self.current_y_scale.set(transform['y_scale'])
                self.current_z_scale.set(transform['z_scale'])
                self.current_x_offset.set(transform['x_offset'])
                self.current_y_offset.set(transform['y_offset'])
                self.current_z_offset.set(transform['z_offset'])
                self.pen_offset_x_var.set(transform['pen_x_offset'])
                self.pen_offset_y_var.set(transform['pen_y_offset'])
                self.current_visibility_var.set(file_data['visible'])

        self.update_ui_for_selection()


    def update_ui_for_selection(self):
        """Enables/disables controls based on whether a file is selected."""
        index = self.selected_file_index.get()
        is_selected = (index != -1)
        state = tk.NORMAL if is_selected else tk.DISABLED

        # Transformation entry fields
        self.rotation_entry.config(state=state) # Enable/disable rotation entry
        self.scale_x_entry.config(state=state)
        self.scale_y_entry.config(state=state)
        self.scale_z_entry.config(state=state)
        self.offset_x_entry.config(state=state)
        self.offset_y_entry.config(state=state)
        self.offset_z_entry.config(state=state)
        self.pen_offset_x_entry.config(state=state)
        self.pen_offset_y_entry.config(state=state)

        # Buttons acting on selected file
        self.apply_transforms_button.config(state=state)
        self.remove_button.config(state=state)
        self.visible_check.config(state=state)
        self.color_button.config(state=state)
        self.reset_button.config(state=state)
        self.center_button.config(state=state)
        self.ender3_offset_button.config(state=state)

        # Global buttons needing > 0 files
        has_files = bool(self.gcode_files)
        global_state = tk.NORMAL if has_files else tk.DISABLED
        self.clear_all_button.config(state=global_state)
        self.save_button.config(state=global_state)

    def clear_transform_entries(self):
        """Sets transform vars to default when no file is selected."""
        self.rotation_angle_var.set(0.0) # Reset rotation angle
        self.current_x_scale.set(1.0)
        self.current_y_scale.set(1.0)
        self.current_z_scale.set(1.0)
        self.current_x_offset.set(0.0)
        self.current_y_offset.set(0.0)
        self.current_z_offset.set(0.0)
        self.pen_offset_x_var.set(0.0)
        self.pen_offset_y_var.set(0.0)
        self.current_visibility_var.set(True) # Default visibility display


    def remove_selected_file(self):
        """Removes the currently selected file from the list and data."""
        index = self.selected_file_index.get()
        if 0 <= index < len(self.gcode_files):
            removed_name = self.gcode_files[index]['basename']
            del self.gcode_files[index]
            self.file_listbox.delete(index)

            # Adjust IDs of subsequent files if needed (optional, depends if ID used heavily)
            for i in range(index, len(self.gcode_files)):
                self.gcode_files[i]['id'] = i # Re-index

            self.selected_file_index.set(-1) # Clear selection
            self.clear_transform_entries()
            self.update_ui_for_selection()
            self.redraw_canvas()
            self.update_status_files_loaded()
            self.status_label.config(text=f"Removed file: {removed_name}")
        else:
            self.status_label.config(text="No file selected to remove.")


    def reset_app_state(self):
        """Clears all loaded files and resets the application state."""
        self.gcode_files = []
        self.file_listbox.delete(0, tk.END)
        self.selected_file_index.set(-1)
        self.clear_transform_entries()
        self.update_ui_for_selection() # Disables buttons
        self.paper_type_var.set("None") # Reset paper
        self.paper_offset_x_var.set(0.0)
        self.paper_offset_y_var.set(0.0)
        self.clear_image_data() # Clear image stuff too
        self.canvas.delete("gcode_path", "bounds_rect", "paper_overlay") # Keep bed
        self.update_bed_display_params()
        self.draw_bed() # Redraw bed based on current params
        self.canvas.configure(scrollregion=self.canvas.bbox("bed"))
        self.update_status_files_loaded()
        self.bounds_label.config(text="Visible Bounds: N/A")
        self.status_label.config(text="Cleared all files and image.")

    def update_status_files_loaded(self):
         self.files_loaded_label.config(text=f"Files loaded: {len(self.gcode_files)}")

    # --- Transformations for Selected File ---
    # (apply_transformations_to_selected, reset_selected_file_transforms, center_selected_on_bed,
    #  apply_ender3_offset_to_selected remain unchanged)
    def apply_transformations_to_selected(self):
        """Reads entries, updates selected file's transform dict, recalculates its segments, redraws."""
        index = self.selected_file_index.get()
        if not (0 <= index < len(self.gcode_files)):
            self.status_label.config(text="No file selected to apply transformations.")
            return

        file_data = self.gcode_files[index]
        try:
            # Read values from the DoubleVars (which are linked to entries)
            transform = file_data['transform'] # Get reference to the dict
            # Add rotation to the transform dictionary
            transform['rotation_angle'] = self.rotation_angle_var.get()
            transform['x_scale'] = self.current_x_scale.get()
            transform['y_scale'] = self.current_y_scale.get()
            transform['z_scale'] = self.current_z_scale.get()
            transform['x_offset'] = self.current_x_offset.get()
            transform['y_offset'] = self.current_y_offset.get()
            transform['z_offset'] = self.current_z_offset.get()
            transform['pen_x_offset'] = self.pen_offset_x_var.get()
            transform['pen_y_offset'] = self.pen_offset_y_var.get()

            # Recalculate current_segments for *this file only*
            file_data['current_segments'] = apply_transformations_to_coords(
                file_data['original_segments'], transform
            )

            self.redraw_canvas() # Redraw everything
            self.status_label.config(text=f"Transforms applied to: {file_data['basename']}")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for scale/offset/pen.")
            self.status_label.config(text="Invalid input.")
        except Exception as e:
             messagebox.showerror("Error", f"Transformation error:\n{e}")
             self.status_label.config(text="Transformation error.")

    def reset_selected_file_transforms(self):
        """Resets the transformations for the currently selected file to defaults."""
        index = self.selected_file_index.get()
        if 0 <= index < len(self.gcode_files):
            file_data = self.gcode_files[index]
            # Reset the transform dictionary
            file_data['transform'] = {
                'x_scale': 1.0, 'y_scale': 1.0, 'z_scale': 1.0,
                'x_offset': 0.0, 'y_offset': 0.0, 'z_offset': 0.0,
                'pen_x_offset': 0.0, 'pen_y_offset': 0.0
            }
            # Update the display variables/entries
            self.on_file_select() # Easiest way to refresh the UI vars
            # Apply the reset (recalculate segments and redraw)
            self.apply_transformations_to_selected()
            self.status_label.config(text=f"Reset transforms for: {file_data['basename']}")
        else:
            self.status_label.config(text="No file selected to reset.")


    def center_selected_on_bed(self):
        """Centers the selected file on the bed by adjusting its offset."""
        index = self.selected_file_index.get()
        if not (0 <= index < len(self.gcode_files)):
            self.status_label.config(text="No file selected to center."); return

        file_data = self.gcode_files[index]
        transform = file_data['transform'] # Get current transform

        try:
            # Get relevant current parameters
            x_s, y_s = transform['x_scale'], transform['y_scale']
            pen_x, pen_y = transform['pen_x_offset'], transform['pen_y_offset']
            bed_w, bed_h = self.bed_width_var.get(), self.bed_height_var.get()

            # Calculate bounds of the file's *original* segments scaled and pen-offset, but with ZERO primary offset
            temp_segments = apply_transformations_to_coords(
                file_data['original_segments'],
                {'x_scale': x_s, 'y_scale': y_s, 'x_offset': 0, 'y_offset': 0, # Use 0 offset here
                 'pen_x_offset': pen_x, 'pen_y_offset': pen_y, 'z_scale': 1, 'z_offset': 0} # Z irrelevant for bounds
            )
            if not temp_segments: self.status_label.config(text="Selected file has no plottable data."); return

            min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
            for segment in temp_segments:
                for point in segment:
                    min_x, max_x = min(min_x, point['x']), max(max_x, point['x'])
                    min_y, max_y = min(min_y, point['y']), max(max_y, point['y'])

            # Handle case where file has no extent (e.g., single point)
            if math.isinf(min_x): self.status_label.config(text="Cannot center file with no plottable points."); return

            # Calculate required primary offset
            path_width = max_x - min_x
            path_height = max_y - min_y
            path_center_x = min_x + path_width / 2.0
            path_center_y = min_y + path_height / 2.0
            bed_center_x = bed_w / 2.0
            bed_center_y = bed_h / 2.0
            new_x_offset = bed_center_x - path_center_x
            new_y_offset = bed_center_y - path_center_y

            # Update the offset vars and apply
            self.current_x_offset.set(round(new_x_offset, 6))
            self.current_y_offset.set(round(new_y_offset, 6))
            self.apply_transformations_to_selected() # Applies using the updated vars
            self.status_label.config(text=f"Centered: {file_data['basename']}")

        except ValueError: messagebox.showerror("Invalid Input", "Invalid scale, pen offset, or bed dimensions."); return
        except Exception as e: messagebox.showerror("Error", f"Centering error: {e}")


    def apply_ender3_offset_to_selected(self):
        """Adjusts selected file's offset so its min X/Y corner is at (10, 10)."""
        index = self.selected_file_index.get()
        if not (0 <= index < len(self.gcode_files)):
            self.status_label.config(text="No file selected."); return

        file_data = self.gcode_files[index]

        # Calculate bounds using the file's *current* transformations
        current_bounds = self.get_bounds_for_file(index)
        if not current_bounds: self.status_label.config(text="Selected file has no plottable data."); return

        target_min_x, target_min_y = 10.0, 10.0
        try:
            # Calculate *additional* primary offset needed
            current_x_offset = file_data['transform']['x_offset']
            current_y_offset = file_data['transform']['y_offset']
            offset_needed_x = target_min_x - current_bounds['min_x']
            offset_needed_y = target_min_y - current_bounds['min_y']

            # Set the new offset in the DoubleVars and apply
            self.current_x_offset.set(round(current_x_offset + offset_needed_x, 6))
            self.current_y_offset.set(round(current_y_offset + offset_needed_y, 6))
            self.apply_transformations_to_selected()
            self.status_label.config(text=f"Margin set for: {file_data['basename']}")
        except Exception as e:
            messagebox.showerror("Error", f"Margin error: {e}")


    # --- Visibility and Color ---
    # (toggle_visibility, change_color remain unchanged)
    def toggle_visibility(self):
        """Toggles the visibility flag of the selected file and redraws."""
        index = self.selected_file_index.get()
        if 0 <= index < len(self.gcode_files):
            file_data = self.gcode_files[index]
            file_data['visible'] = self.current_visibility_var.get()
            self.redraw_canvas()
            self.status_label.config(text=f"Visibility updated for: {file_data['basename']}")


    def change_color(self):
        """Opens a color chooser to change the selected file's path color."""
        index = self.selected_file_index.get()
        if 0 <= index < len(self.gcode_files):
            file_data = self.gcode_files[index]
            initial_color = file_data['color']
            # askcolor returns tuple ((r,g,b), hexstring) or (None, None)
            color_tuple = colorchooser.askcolor(color=initial_color, title=f"Choose color for {file_data['basename']}")
            if color_tuple and color_tuple[1]: # Check if a color was chosen (hexstring is not None)
                file_data['color'] = color_tuple[1] # Store the hex string
                self.redraw_canvas()
                self.status_label.config(text=f"Color updated for: {file_data['basename']}")


    # --- Drawing ---

    def draw_bed(self):
        """Draws the bed outline, grid, and origin marker."""
        self.canvas.delete("bed")
        if not hasattr(self, 'bed_cx_min'):
            if not self.update_bed_display_params(): return # Don't draw if params invalid

        # Draw bed rectangle
        self.canvas.create_rectangle(
            self.bed_cx_min, self.bed_cy_min, self.bed_cx_max, self.bed_cy_max,
            outline="darkgray", width=2, tags=("bed", "bed_visual")
        )

        # Draw grid lines
        grid_spacing_mm = 10
        # Avoid potential errors if bed dimensions are zero or negative temporarily
        if self.bed_width_mm > 0 and self.bed_height_mm > 0:
            num_x = max(0, int(self.bed_width_mm // grid_spacing_mm))
            num_y = max(0, int(self.bed_height_mm // grid_spacing_mm))
            
            # Calculate exact grid spacing in canvas pixels to match the image scaling
            grid_spacing_px = grid_spacing_mm * self.canvas_scale
            
            # Draw X grid lines
            for i in range(1, num_x + 1):
                # Use consistent scaling for grid lines
                cx = self.bed_cx_min + (i * grid_spacing_px)
                self.canvas.create_line(cx, self.bed_cy_min, cx, self.bed_cy_max, fill="lightgray", tags=("bed", "bed_visual"))
            
            # Draw Y grid lines
            for i in range(1, num_y + 1):
                # Use consistent scaling for grid lines
                cy = self.bed_cy_max - (i * grid_spacing_px)
                self.canvas.create_line(self.bed_cx_min, cy, self.bed_cx_max, cy, fill="lightgray", tags=("bed", "bed_visual"))

        # Draw origin marker
        cx0, cy0 = self.gcode_to_canvas(0, 0)
        self.canvas.create_oval(cx0-3, cy0-3, cx0+3, cy0+3, fill="red", outline="red", tags=("bed", "bed_visual"))
        self.canvas.create_text(cx0 + 5, cy0 + 5, text="(0,0)", anchor="nw", fill="red", tags=("bed", "bed_visual"), font=("Arial", 8))

        self.canvas.tag_lower("bed_visual") # Ensure grid/origin are behind everything else initially


    def draw_paper_overlay(self):
        """Draws the selected paper size at the specified offset."""
        self.canvas.delete("paper_overlay")
        paper_type = self.paper_type_var.get()
        if (paper_type == "None" or paper_type not in PAPER_SIZES):
            return
        try:
            offset_x = self.paper_offset_x_var.get()
            offset_y = self.paper_offset_y_var.get()
            paper_dims = PAPER_SIZES[paper_type]
            if paper_dims is None: return # Should be caught by "None" check, but safety
            width, height = paper_dims

            # Calculate corners in G-code coordinates
            g_min_x, g_min_y = offset_x, offset_y
            g_max_x, g_max_y = offset_x + width, offset_y + height

            # Convert corners to canvas coordinates
            c_min_x, c_max_y = self.gcode_to_canvas(g_min_x, g_min_y) # Bottom-left GCode -> Top-left Canvas (?) No, BL GCode -> TL Canvas
            c_max_x, c_min_y = self.gcode_to_canvas(g_max_x, g_max_y) # Top-right GCode -> Bottom-right Canvas (?) No, TR GCode -> BR Canvas

            # Correcting conversion understanding:
            # gcode_to_canvas converts (GCodeX, GCodeY) -> (CanvasX, CanvasY)
            # GCode Origin: Bottom-Left (X increases right, Y increases up)
            # Canvas Origin: Top-Left (X increases right, Y increases down)
            # gcode_to_canvas handles the Y flip.
            # Paper Bottom-Left (GCode): (offset_x, offset_y) -> Canvas: (c_bl_x, c_bl_y)
            # Paper Top-Right (GCode): (offset_x + width, offset_y + height) -> Canvas: (c_tr_x, c_tr_y)
            # Canvas rectangle needs top-left and bottom-right canvas coordinates.
            # Top-Left Canvas coord corresponds to Bottom-Left GCode coord because Y is flipped.
            # Bottom-Right Canvas coord corresponds to Top-Right GCode coord because Y is flipped.

            # Calculate canvas coords for all 4 corners of the paper
            c_bl_x, c_bl_y = self.gcode_to_canvas(g_min_x, g_min_y)   # Paper Bottom-Left
            c_tl_x, c_tl_y = self.gcode_to_canvas(g_min_x, g_max_y)   # Paper Top-Left
            c_tr_x, c_tr_y = self.gcode_to_canvas(g_max_x, g_max_y)   # Paper Top-Right
            c_br_x, c_br_y = self.gcode_to_canvas(g_max_x, g_min_y)   # Paper Bottom-Right

            # Canvas rectangle uses top-left and bottom-right canvas points
            # Top-Left Canvas point is (c_tl_x, c_tl_y)
            # Bottom-Right Canvas point is (c_br_x, c_br_y)

            # Draw the rectangle using canvas coordinates
            paper_id = self.canvas.create_rectangle(
                c_tl_x, c_tl_y, # Top-Left Canvas
                c_br_x, c_br_y, # Bottom-Right Canvas
                outline="purple", dash=(4, 4), width=1, tags="paper_overlay"
            )

            # Order matters: Raise paper above bed, but below gcode/bounds
            if self.canvas.find_withtag("bed_visual"):
                self.canvas.tag_raise(paper_id, "bed_visual")
            # Lowering relative to specific tags ensures it's behind them if they exist
            if self.canvas.find_withtag("gcode_path"):
                self.canvas.tag_lower(paper_id, "gcode_path")
            if self.canvas.find_withtag("bounds_rect"):
                self.canvas.tag_lower(paper_id, "bounds_rect")

        except ValueError: self.status_label.config(text="Invalid paper offset value.")
        except Exception as e: self.status_label.config(text=f"Error drawing paper: {e}")

    def on_paper_type_change(self, event=None):
        # Redrawing canvas ensures correct layering
        self.redraw_canvas()

    def update_paper_overlay_action(self):
        # Redrawing canvas ensures correct layering
        self.redraw_canvas()
        self.status_label.config(text="Paper overlay updated.")


    # *** ADDED MISSING draw_gcode METHOD ***
    def draw_gcode(self):
        """Draws the G-code paths for all visible files and the combined bounds."""
        self.canvas.delete("gcode_path", "bounds_rect")

        hide_z = self.hide_z_moves_var.get()
        any_visible = False

        for index, file_data in enumerate(self.gcode_files):
            if not file_data['visible']: continue
            any_visible = True

            file_tag = f"gcode_file_{index}"
            color = file_data['color']
            segments = file_data['current_segments']

            for segment in segments:
                # Check if the segment is just a Z-only move (start and end points have same X/Y)
                # NOTE: The parser already adds a 'z_only' flag to the *endpoint* of a Z-only move.
                # G0 segments are handled differently (drawn dashed).
                # G1 segments ending in a Z-only move mark the end of a plotting line.
                # A segment consisting of two points where the first is a G1 start and the second is z_only=True
                # represents a Z-only move *after* potentially plotting.

                is_g0_segment = (len(segment) == 2 and segment[0]['cmd'] == "G0_START" and segment[1]['cmd'] == "G0")

                if is_g0_segment:
                    if hide_z: continue # Hide all G0 travel moves if checked
                    # Draw G0 moves as dashed lines
                    p1_x, p1_y = self.gcode_to_canvas(segment[0]['x'], segment[0]['y'])
                    p2_x, p2_y = self.gcode_to_canvas(segment[1]['x'], segment[1]['y'])
                    self.canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill=color, width=1,
                                            dash=(3, 3), tags=("gcode_path", file_tag, "g0_move"))
                else: # It's a G1 segment (or potentially includes Z-only points within it)
                    # Draw G1 moves as solid lines, potentially skipping z-only parts if hidden
                    canvas_points = []
                    last_point_was_z_only = False
                    for i, point in enumerate(segment):
                        # The 'z_only' flag in the point data indicates *this specific G1 move*
                        # was only in Z, not whether the *entire segment* is just Z.
                        # We want to hide segments where *only* Z changed. The parser logic
                        # tries to handle this by ending segments on Z moves, but G0 moves
                        # are the main target for hiding 'travel'. We will keep hide_z logic
                        # simple and only apply it to G0 for now.

                        # Add point to the list for line drawing
                        cx, cy = self.gcode_to_canvas(point['x'], point['y'])
                        canvas_points.extend([cx, cy])

                    if len(canvas_points) >= 4: # Need at least two points (x1,y1,x2,y2)
                        self.canvas.create_line(*canvas_points, fill=color, width=1.5, # Thicker lines for G1
                                                tags=("gcode_path", file_tag, "g1_move"))

        # Calculate and draw bounds rectangle for visible files
        current_bounds = self.get_current_bounds() # Gets bounds of all visible files
        if current_bounds:
            min_cx, max_cy = self.gcode_to_canvas(current_bounds["min_x"], current_bounds["min_y"])
            max_cx, min_cy = self.gcode_to_canvas(current_bounds["max_x"], current_bounds["max_y"])

            # Draw bounding box
            self.canvas.create_rectangle(min_cx, min_cy, max_cx, max_cy,
                                         outline="gray", dash=(2, 2), tags=("gcode_path", "bounds_rect"))

            # Update bounds label
            width = current_bounds["max_x"] - current_bounds["min_x"]
            height = current_bounds["max_y"] - current_bounds["min_y"]
            bounds_text = f"Visible Bounds: X[{current_bounds['min_x']:.2f} .. {current_bounds['max_x']:.2f}] Y[{current_bounds['min_y']:.2f} .. {current_bounds['max_y']:.2f}] (W:{width:.2f} H:{height:.2f})"
            self.bounds_label.config(text=bounds_text)
        elif any_visible:
            self.bounds_label.config(text="Visible Bounds: Empty or single point.")
        else:
            self.bounds_label.config(text="Visible Bounds: N/A (No files visible)")


    def draw_warped_image(self):
        """Draws the warped bed image on the canvas background if available and enabled."""
        self.canvas.delete("warped_bed_image") # Remove previous instance

        if self.show_warped_image_var.get() and self.warped_bed_image_pil and hasattr(self, 'canvas_scale'):
            try:
                # Ensure bed params are valid (calls update_bed_display_params if needed)
                if not hasattr(self, 'bed_cx_min'):
                    if not self.update_bed_display_params():
                         print("Warning: Bed parameters invalid, cannot draw warped image.")
                         return

                # Use the actual bed dimensions in canvas coordinates for exact placement
                # This ensures the image perfectly lines up with the bed grid
                img_width = self.bed_cx_max - self.bed_cx_min
                img_height = self.bed_cy_max - self.bed_cy_min

                # Resize the canonical PIL image for display (use NEAREST for speed)
                resized_pil = self.warped_bed_image_pil.resize(
                    (max(1, int(img_width)), max(1, int(img_height))), 
                    Image.Resampling.NEAREST
                )
                
                # Create/update the Tkinter PhotoImage
                self.warped_bed_image_tk = ImageTk.PhotoImage(resized_pil)

                # Draw the image using create_image with image fill
                # This makes sure the image stretches exactly between the bed corners
                self.canvas.create_image(
                    self.bed_cx_min, self.bed_cy_min,
                    anchor=tk.NW,
                    image=self.warped_bed_image_tk,
                    tags="warped_bed_image"
                )
                
                # Ensure it's the absolute bottom layer
                self.canvas.tag_lower("warped_bed_image")

            except Exception as e:
                print(f"Error drawing warped image: {e}")
                self.canvas.delete("warped_bed_image") # Clean up if error occurs
                self.warped_bed_image_tk = None # Reset tk image on error


    def redraw_canvas(self):
        """Redraws everything: Warped Image (optional), Bed, Paper, G-code, updates scroll."""
        self.canvas.delete("all") # Clear everything before full redraw

        # Draw layers from bottom up
        self.draw_warped_image()     # Bottom layer (optional background)
        self.draw_bed()              # Bed grid/outline/origin
        self.draw_paper_overlay()    # Virtual paper
        self.draw_gcode()            # G-code paths and combined bounds rect

        # Redraw temporary items if active
        self.redraw_temp_image_display()
        self.redraw_corner_markers()

        self.update_scroll_region()


    def update_scroll_region(self):
         """Updates the canvas scrollregion to fit all drawn items."""
         try:
             # Consider only visual elements that define the extent
             bbox = self.canvas.bbox("bed_visual", "paper_overlay", "gcode_path", "bounds_rect", "temp_image_display")
             if bbox:
                 # Add padding around the bounding box
                 pad = 50
                 padded_bbox = (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad)
                 self.canvas.configure(scrollregion=padded_bbox)
             else:
                 # Fallback to initial view size if nothing is drawn
                 # Use bed outline if it exists, otherwise the configured size
                 bed_bbox = self.canvas.bbox("bed_visual")
                 if bed_bbox:
                      pad = 50
                      padded_bbox = (bed_bbox[0]-pad, bed_bbox[1]-pad, bed_bbox[2]+pad, bed_bbox[3]+pad)
                      self.canvas.configure(scrollregion=padded_bbox)
                 else:
                      self.canvas.configure(scrollregion=(0, 0, self.canvas_view_width, self.canvas_view_height))
         except Exception as e:
              print(f"Error updating scroll region: {e}")
              # Fallback in case of error
              self.canvas.configure(scrollregion=(0, 0, self.canvas_view_width, self.canvas_view_height))


    # --- Bounds Calculation ---
    # (get_bounds_for_file, get_current_bounds remain unchanged)
    def get_bounds_for_file(self, index):
        """Calculates bounds for a specific file index based on its current_segments."""
        if not (0 <= index < len(self.gcode_files)): return None
        file_data = self.gcode_files[index]
        if not file_data['current_segments']: return None

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        has_points = False
        for segment in file_data['current_segments']:
            for point in segment:
                 # Exclude points from pure Z moves if hide_z is checked? No, bounds should reflect full extent.
                has_points = True
                min_x, max_x = min(min_x, point['x']), max(max_x, point['x'])
                min_y, max_y = min(min_y, point['y']), max(max_y, point['y'])
        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y} if has_points else None


    def get_current_bounds(self):
        """Calculates combined bounds for ALL VISIBLE files."""
        overall_min_x, overall_min_y = float('inf'), float('inf')
        overall_max_x, overall_max_y = float('-inf'), float('-inf')
        has_any_points = False

        for index, file_data in enumerate(self.gcode_files):
            if not file_data['visible']: continue

            # Use pre-calculated segments for the file
            segments = file_data['current_segments']
            if not segments: continue

            for segment in segments:
                 for point in segment:
                    # Exclude points from pure Z moves if hide_z is checked? No, bounds should reflect full extent.
                    has_any_points = True
                    overall_min_x = min(overall_min_x, point['x'])
                    overall_max_x = max(overall_max_x, point['x'])
                    overall_min_y = min(overall_min_y, point['y'])
                    overall_max_y = max(overall_max_y, point['y'])

        if has_any_points:
            return {"min_x": overall_min_x, "max_x": overall_max_x, "min_y": overall_min_y, "max_y": overall_max_y}
        else:
            return None


    # --- Mouse Handlers (Panning, Dragging adapted) ---

    def on_pan_start(self, event):
        # Prevent panning if in corner selection mode
        if self.corner_selection_mode: return
        self.canvas.scan_mark(event.x, event.y)
        self._pan_data["x"] = event.x; self._pan_data["y"] = event.y
        self.canvas.config(cursor="fleur")

    def on_pan_motion(self, event):
        if self.corner_selection_mode: return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_stop(self, event):
        if self.corner_selection_mode: return
        self.canvas.config(cursor="")

    def on_mouse_wheel(self, event):
        """Basic zoom centered on the mouse cursor."""
        if self.corner_selection_mode: return # Disable zoom during corner selection

        # Determine scale factor based on platform and event type
        scale_factor = 1.0
        # Windows/macOS use event.delta
        if hasattr(event, 'delta'):
            if event.delta > 0:
                scale_factor = 1.1
            elif event.delta < 0:
                scale_factor = 1 / 1.1
        # Linux uses event.num (Button-4 for scroll up, Button-5 for scroll down)
        elif hasattr(event, 'num'):
            if event.num == 4:
                scale_factor = 1.1
            elif event.num == 5:
                scale_factor = 1 / 1.1
        else: # Should not happen for standard mouse wheels
            return

        # Get canvas coordinates under the mouse
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Scale the canvas - this scales everything uniformly
        self.canvas.scale("all", cx, cy, scale_factor, scale_factor)
        
        # Update the canvas_scale value
        if hasattr(self, 'canvas_scale'):
            self.canvas_scale *= scale_factor
        
        # Update scroll region to account for the new scale
        self.update_scroll_region()


    def on_drag_start(self, event):
        # Prevent dragging G-code if in corner selection mode
        if self.corner_selection_mode: return

        # Find the topmost item under cursor with a file-specific tag
        # Increase search tolerance slightly for easier clicking
        search_radius = 3
        items = self.canvas.find_overlapping(event.x - search_radius, event.y - search_radius,
                                             event.x + search_radius, event.y + search_radius)
        dragged_file_index = -1
        for item in reversed(items): # Check topmost first
            tags = self.canvas.gettags(item)
            if "gcode_path" in tags and "bed_visual" not in tags and "paper_overlay" not in tags and "bounds_rect" not in tags: # Ensure it's gcode, not bed/paper/bounds
                for tag in tags:
                    if tag.startswith("gcode_file_"):
                        try:
                            potential_index = int(tag.split("_")[-1])
                            # Check if this file actually exists and is visible (can only drag visible files)
                            if 0 <= potential_index < len(self.gcode_files) and self.gcode_files[potential_index]['visible']:
                                dragged_file_index = potential_index
                                break # Found valid file index
                            else:
                                continue # Index invalid or file hidden
                        except (ValueError, IndexError):
                            continue
                if dragged_file_index != -1:
                    break # Stop searching items once a valid gcode file is found

        if dragged_file_index != -1:
            # Select the dragged file in the listbox
            if self.selected_file_index.get() != dragged_file_index:
                 self.file_listbox.selection_clear(0, tk.END)
                 self.file_listbox.selection_set(dragged_file_index)
                 self.file_listbox.see(dragged_file_index)
                 # self.selected_file_index.set(dragged_file_index) # on_file_select will handle this
                 self.on_file_select() # Update UI controls to show the selected file's state

            # Now start the drag state for this specific file
            file_data = self.gcode_files[dragged_file_index]
            self._drag_data["is_dragging"] = True
            self._drag_data["dragged_file_index"] = dragged_file_index
            # Convert event coords to canvas coords for delta calculation
            self._drag_data["last_event_x"] = self.canvas.canvasx(event.x)
            self._drag_data["last_event_y"] = self.canvas.canvasy(event.y)
            # Store the starting primary offset of the *dragged* file
            self._drag_data["start_offset_x"] = file_data['transform']['x_offset']
            self._drag_data["start_offset_y"] = file_data['transform']['y_offset']
            self.canvas.config(cursor="hand2")
            self.status_label.config(text=f"Dragging: {file_data['basename']}")
        else:
            # Clicked on canvas but not on a draggable G-code path
            self._drag_data["is_dragging"] = False
            self._drag_data["dragged_file_index"] = -1
            # Also call the general release handler in case a corner click was intended but missed
            # It's better to handle this in on_general_click/release or on_drag_stop
            # self.on_general_click(event) # Avoid calling this here


    def on_drag_motion(self, event):
        # Do nothing if corner selection is active
        if self.corner_selection_mode: return

        if not self._drag_data["is_dragging"]: return

        index = self._drag_data["dragged_file_index"]
        if not (0 <= index < len(self.gcode_files)):
            self._drag_data["is_dragging"] = False # Safety check
            return

        # Convert current event coords to canvas coords
        current_canvas_x = self.canvas.canvasx(event.x)
        current_canvas_y = self.canvas.canvasy(event.y)

        # Calculate delta in canvas pixels
        delta_cx = current_canvas_x - self._drag_data["last_event_x"]
        delta_cy = current_canvas_y - self._drag_data["last_event_y"]

        # Convert pixel delta to G-code offset delta
        # We need the canvas scale factor AT THE TIME OF DRAGGING
        # which depends on the current zoom level.
        # Re-calculating canvas_scale might be complex if zoom happened during drag.
        # Instead, convert the start and end canvas points to gcode points.
        start_gx, start_gy = self.canvas_to_gcode(self._drag_data["last_event_x"], self._drag_data["last_event_y"])
        current_gx, current_gy = self.canvas_to_gcode(current_canvas_x, current_canvas_y)

        delta_gx = current_gx - start_gx
        delta_gy = current_gy - start_gy

        # Calculate the new *total* primary offset for the dragged file
        # Use the original starting offset stored in _drag_data plus the total gcode delta since drag started
        # NOTE: The previous implementation updated start_offset continuously, which is also valid.
        # Let's stick to the simpler approach: Total Offset = Initial Offset + (Current Gcode Pos - Initial Gcode Pos)

        # Alternative: Use the effective start offset from the previous motion event (current approach)
        new_offset_x = self._drag_data["start_offset_x"] + delta_gx
        new_offset_y = self._drag_data["start_offset_y"] + delta_gy


        # --- Update Dragged File's State ---
        file_data = self.gcode_files[index]
        # 1. Update the file's transform dictionary directly
        file_data['transform']['x_offset'] = new_offset_x
        file_data['transform']['y_offset'] = new_offset_y

        # 2. Update the linked Tkinter Variables (immediately updates entry fields)
        # Check if the currently *selected* file is the one being dragged
        if self.selected_file_index.get() == index:
            self.current_x_offset.set(round(new_offset_x, 6))
            self.current_y_offset.set(round(new_offset_y, 6))

        # 3. Recalculate segments for the dragged file
        file_data['current_segments'] = apply_transformations_to_coords(
            file_data['original_segments'], file_data['transform']
        )

        # --- Redraw for Preview ---
        # Redrawing only the affected file + bounds is complex with current setup.
        # Redraw everything for simplicity, optimize if performance suffers significantly.
        self.redraw_canvas()

        # Update last canvas coords and effective starting offset for next delta calculation
        self._drag_data["last_event_x"] = current_canvas_x
        self._drag_data["last_event_y"] = current_canvas_y
        # Update the effective start offset based on the G-code coordinates just applied
        self._drag_data["start_offset_x"] = new_offset_x
        self._drag_data["start_offset_y"] = new_offset_y

        # Update status
        self.status_label.config(text=f"Dragging {file_data['basename']}... Offset X:{new_offset_x:.2f} Y:{new_offset_y:.2f}")


    def on_drag_stop(self, event):
        if self._drag_data["is_dragging"]:
            index = self._drag_data["dragged_file_index"]
            self.canvas.config(cursor="")
            self._drag_data["is_dragging"] = False
            # Don't reset dragged_file_index immediately, might be needed below

            # Final state is already set by the last on_drag_motion update
            # and redraw_canvas was called there. Just update status.
            if 0 <= index < len(self.gcode_files):
                 # Ensure the UI entry fields accurately reflect the final dragged state
                 # especially if the dragged file wasn't the selected one initially
                 if self.selected_file_index.get() == index:
                      file_data = self.gcode_files[index]
                      # Round values before setting to avoid potential float precision issues in display
                      final_x = round(file_data['transform']['x_offset'], 6)
                      final_y = round(file_data['transform']['y_offset'], 6)
                      self.current_x_offset.set(final_x)
                      self.current_y_offset.set(final_y)
                      # Update the transform dict with the rounded value as well
                      file_data['transform']['x_offset'] = final_x
                      file_data['transform']['y_offset'] = final_y
                 self.status_label.config(text=f"Drag complete for: {self.gcode_files[index]['basename']}")
            else:
                 self.status_label.config(text="Drag complete.")
            self._drag_data["dragged_file_index"] = -1 # Reset index now
        else:
            # If not dragging, treat release as a general release (might finalize corner click)
            self.on_general_release(event)

        # Ensure reset even if something went wrong
        self._drag_data["is_dragging"] = False
        if self._drag_data["dragged_file_index"] != -1: # Only reset if it wasn't already
             self._drag_data["dragged_file_index"] = -1


    def on_general_click(self, event):
        """Handles clicks NOT consumed by dragging gcode paths."""
        # Check if corner selection is active AND the click binding is set
        # This prevents handling general clicks if the specific corner binding is active
        if self.corner_selection_mode and self._corner_click_binding_id:
             # Let the on_corner_click handle this via its binding
             pass
        # Add other general click actions here if needed
        # E.g., deselect file if clicking on empty space? Maybe not desired.


    def on_general_release(self, event):
        """Handles button releases NOT consumed by dragging gcode paths."""
        # Could be used to finalize selection modes, etc.
        pass # Nothing specific needed here for now


    # --- NEW: Bed Image and Warping Methods ---

    def update_image_controls_state(self):
        """Enable/disable image controls based on image loaded state and corner selections."""
        if not IMAGE_PROCESSING_ENABLED: return # Keep disabled if libs missing

        image_loaded = self.original_bed_image_pil is not None
        bed_corners_set = len(self.bed_image_corners_src) == 4
        paper_corners_set = len(self.paper_image_corners_src) == 4 # Check length

        state_if_img_loaded = tk.NORMAL if image_loaded else tk.DISABLED
        self.select_bed_corners_button.config(state=state_if_img_loaded)
        # Only enable paper corners if bed corners are also set (as paper warp depends on bed warp conceptually)
        # Or, allow paper selection independently? For now, let's allow independent paper selection
        self.select_paper_corners_button.config(state=state_if_img_loaded)
        self.clear_image_button.config(state=state_if_img_loaded)

        # Enable "Show Warped" only if warp is possible (image loaded and bed corners selected)
        can_warp = image_loaded and bed_corners_set and self.warped_bed_image_pil is not None
        self.show_warped_image_check.config(state=tk.NORMAL if can_warp else tk.DISABLED)
        if not can_warp:
            self.show_warped_image_var.set(False) # Ensure it's off if warp isn't possible/done

        # Update status label
        if image_loaded:
             # Use len() directly on the lists
             bed_status = f"{len(self.bed_image_corners_src)}/4 Bed"
             paper_status = f"{len(self.paper_image_corners_src)}/4 Paper"
             self.corner_status_label.config(text=f"Corners selected: {bed_status}, {paper_status}")
        else:
             self.corner_status_label.config(text="Load image first")


    def load_bed_image(self):
        """Loads an image file for the bed background."""
        if not IMAGE_PROCESSING_ENABLED:
             messagebox.showerror("Missing Libraries", "Cannot load image. Please install OpenCV, NumPy, and Pillow.")
             return
        filepath = filedialog.askopenfilename(
            title="Open Bed Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All Files", "*.*")]
        )
        if not filepath: return

        try:
            # Clear previous image data first, keep PIL momentarily for display if needed
            self.clear_image_data(clear_pil_obj=False) # keep PIL obj for immediate display

            self.original_bed_image_pil = Image.open(filepath)
            # Ensure image is in RGB format for consistency before converting to BGR
            if self.original_bed_image_pil.mode != 'RGB':
                 self.original_bed_image_pil = self.original_bed_image_pil.convert('RGB')

            # Convert to OpenCV format (NumPy array, BGR color)
            self.original_bed_image_cv = cv2.cvtColor(np.array(self.original_bed_image_pil), cv2.COLOR_RGB2BGR)

            self.status_label.config(text=f"Loaded image: {os.path.basename(filepath)}")
            # Display the image immediately *after* loading and converting
            self.display_temp_original_image()
            self.update_image_controls_state() # Enable selection buttons etc.

        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not load or process image:\n{filepath}\n\n{e}")
            self.clear_image_data() # Ensure clean state on error
            self.status_label.config(text="Error loading image.")


    def display_temp_original_image(self):
        """Displays the original loaded image temporarily on the canvas for selection."""
        if not self.original_bed_image_pil: return
        self.canvas.delete("temp_image_display") # Remove old one if exists

        img = self.original_bed_image_pil
        img_w, img_h = img.size

        # Calculate scale to fit within the current canvas *view* (not scroll region)
        view_w = self.canvas.winfo_width()
        view_h = self.canvas.winfo_height()
        if view_w <= 1 or view_h <= 1 : # Initial state before window is fully drawn
             view_w, view_h = self.canvas_view_width, self.canvas_view_height # Use configured size

        # Add some padding within the view
        target_w = view_w * 0.9
        target_h = view_h * 0.9

        scale_w = target_w / img_w if img_w > 0 else 1.0
        scale_h = target_h / img_h if img_h > 0 else 1.0
        self.image_display_scale_factor = min(scale_w, scale_h, 1.0) # Don't scale up

        # Resize using PIL for display
        disp_w = max(1, int(img_w * self.image_display_scale_factor)) # Ensure width > 0
        disp_h = max(1, int(img_h * self.image_display_scale_factor)) # Ensure height > 0
        try:
             # Use LANCZOS (formerly ANTIALIAS) for better quality resizing
             img_display = img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        except AttributeError: # Handle older Pillow versions
             img_display = img.resize((disp_w, disp_h), Image.ANTIALIAS)
        except ValueError as e: # Catch potential "tile cannot extend outside image"
             messagebox.showerror("Image Resize Error", f"Error resizing image for display:\n{e}\n\nPlease try a different image or check image integrity.")
             self.clear_image_data()
             return

        # Store the PhotoImage on self to prevent garbage collection
        self.temp_original_bed_image_tk = ImageTk.PhotoImage(img_display)

        # Draw in the center of the current view
        # Convert view center to canvas coords using canvasx/y
        center_x = self.canvas.canvasx(view_w / 2)
        center_y = self.canvas.canvasy(view_h / 2)

        # Create image centered on canvas view
        self.temp_image_display_id = self.canvas.create_image(
            center_x, center_y,
            anchor=tk.CENTER,
            image=self.temp_original_bed_image_tk, # Use the stored PhotoImage
            tags="temp_image_display"
        )

        # Lower it below GCode/Bed etc. only if those tags exist
        z_order_tags = ["bed_visual", "paper_overlay", "gcode_path", "bounds_rect", "corner_marker"]
        for tag in z_order_tags:
            if self.canvas.find_withtag(tag):
                self.canvas.tag_lower(self.temp_image_display_id, tag)

        self.status_label.config(text="Image loaded. Select corners.")
        self.update_scroll_region() # Adjust scroll if image is large


    def redraw_temp_image_display(self):
         """Redraws the temp image if it exists (used after canvas clear)."""
         # Check if the underlying tk image object still exists
         if self.temp_image_display_id and hasattr(self, 'temp_original_bed_image_tk') and self.temp_original_bed_image_tk:
              try:
                  # Check if PhotoImage object is still valid
                  self.temp_original_bed_image_tk.width()
                  self.temp_original_bed_image_tk.height()
              except tk.TclError:
                  # The PhotoImage object is gone, maybe due to clear_image_data or other issues
                  self.temp_image_display_id = None
                  self.temp_original_bed_image_tk = None
                  return # Cannot redraw

              # Re-calculate position based on current view center
              view_w = self.canvas.winfo_width()
              view_h = self.canvas.winfo_height()
              if view_w <= 1 or view_h <= 1 :
                  view_w, view_h = self.canvas_view_width, self.canvas_view_height

              center_x = self.canvas.canvasx(view_w / 2)
              center_y = self.canvas.canvasy(view_h / 2)

              # Recreate the image item using the existing PhotoImage object
              self.temp_image_display_id = self.canvas.create_image(
                    center_x, center_y,
                    anchor=tk.CENTER,
                    image=self.temp_original_bed_image_tk, # Use the stored PhotoImage
                    tags="temp_image_display"
              )

              # Ensure layering is correct after redraw
              z_order_tags = ["bed_visual", "paper_overlay", "gcode_path", "bounds_rect", "corner_marker"]
              for tag in z_order_tags:
                  if self.canvas.find_withtag(tag):
                      self.canvas.tag_lower(self.temp_image_display_id, tag)
         else:
             # If the tk image is gone or ID is invalid, nullify the ID
             self.temp_image_display_id = None
             self.temp_original_bed_image_tk = None


    def start_corner_selection(self, mode):
        """Initiates the process of selecting 4 corners ('bed' or 'paper')."""
        if not self.original_bed_image_pil:
            messagebox.showwarning("No Image", "Please load a bed image first.")
            return
        if self.corner_selection_mode:
            messagebox.showwarning("Selection Active", f"Already selecting {self.corner_selection_mode} corners. Cancel or finish first.")
            return

        self.corner_selection_mode = mode
        self.corner_selection_points_temp = []
        self.clear_corner_markers() # Clear markers from previous attempts

        # Ensure the temp image is visible and raise it slightly for easier clicking?
        if not self.temp_image_display_id or not self.canvas.coords(self.temp_image_display_id):
             self.display_temp_original_image()
        if self.temp_image_display_id: # Raise it slightly?
             # self.canvas.tag_raise(self.temp_image_display_id) # Maybe problematic for layers
             pass

        # Bind left click on the CANVAS to the handler
        # We need to bind to the canvas, then check if the click was on the image item
        if self._corner_click_binding_id:
            self.canvas.unbind("<Button-1>", self._corner_click_binding_id)
        # Use specific tag binding if possible, otherwise general canvas binding
        # Binding directly to the image tag might be more robust if layering changes
        # self._corner_click_binding_id = self.canvas.tag_bind("temp_image_display", "<Button-1>", self.on_corner_click)
        # Let's stick with general canvas binding and check item overlap in the handler for now
        self._corner_click_binding_id = self.canvas.bind("<Button-1>", self.on_corner_click)


        # Update instructions
        corner_index = len(self.corner_selection_points_temp)
        self.corner_status_label.config(text=f"Click {CORNER_NAMES[corner_index]} corner for {mode.upper()}")
        self.status_label.config(text=f"Corner Selection Mode: Click {mode.upper()} {CORNER_NAMES[corner_index]}. Right-click to cancel.")
        self.canvas.config(cursor="crosshair")

        # Add right-click cancel binding
        self._corner_cancel_binding_id = self.canvas.bind("<Button-3>", self.cancel_corner_selection_event) # Button-3 for right-click


    def cancel_corner_selection_event(self, event=None):
        """Wrapper to call cancel_corner_selection from an event."""
        self.cancel_corner_selection()

    def cancel_corner_selection(self):
        """Cancels the ongoing corner selection process."""
        if not self.corner_selection_mode: return

        # Unbind handlers
        if self._corner_click_binding_id:
            self.canvas.unbind("<Button-1>", self._corner_click_binding_id)
            self._corner_click_binding_id = None
        if hasattr(self, '_corner_cancel_binding_id') and self._corner_cancel_binding_id:
             self.canvas.unbind("<Button-3>", self._corner_cancel_binding_id)
             self._corner_cancel_binding_id = None


        self.clear_corner_markers()
        # Keep temp image displayed, user might want to reselect soon
        # self.canvas.delete("temp_image_display")
        # self.temp_image_display_id = None

        status_msg = f"{self.corner_selection_mode.capitalize()} corner selection cancelled."
        self.corner_selection_mode = None # Reset mode *before* updating UI state
        self.update_image_controls_state() # Re-enable/disable buttons appropriately
        self.status_label.config(text=status_msg)
        self.canvas.config(cursor="")


    def on_corner_click(self, event):
        """Handles mouse clicks during corner selection mode."""
        if not self.corner_selection_mode:
             # If selection isn't active, but binding still exists somehow, remove it
             if self._corner_click_binding_id:
                  self.canvas.unbind("<Button-1>", self._corner_click_binding_id)
                  self._corner_click_binding_id = None
             # If not selecting corners, pass click to general handler (might start drag)
             # Need to avoid calling on_drag_start directly if button-1 is already bound here
             # Let the normal Button-1 binding handle it if this one does nothing.
             # Find items under the cursor to see if it hit a draggable path
             search_radius = 3
             items = self.canvas.find_overlapping(event.x - search_radius, event.y - search_radius,
                                                 event.x + search_radius, event.y + search_radius)
             is_on_gcode = False
             for item in reversed(items):
                 tags = self.canvas.gettags(item)
                 if "gcode_path" in tags and "bed_visual" not in tags and "paper_overlay" not in tags and "bounds_rect" not in tags:
                     is_on_gcode = True
                     break
             # If the click wasn't on gcode, maybe call a general click handler
             if not is_on_gcode:
                  self.on_general_click(event) # Handle clicks on empty space/bed etc.
             # Otherwise, do nothing and let the tag_bind for "gcode_path" handle the drag start.
             return

        # --- Corner selection is active ---
        # Check if the temporary image display item exists
        if not self.temp_image_display_id or not self.canvas.coords(self.temp_image_display_id):
             self.status_label.config(text="Error: Temporary image display lost. Reload image.")
             self.cancel_corner_selection()
             return

        # Find items under the click using find_overlapping
        search_radius = 5
        overlapping_items = self.canvas.find_overlapping(event.x - search_radius, event.y - search_radius, event.x + search_radius, event.y + search_radius)

        # Check if the temporary image display is among the items clicked
        if not self.temp_image_display_id in overlapping_items:
             self.status_label.config(text="Click inside the displayed image to select corners.")
             return # Click was not on the target image

        # --- Click is on the image ---
        # Get click coordinates relative to the CANVAS
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Get the bounding box of the displayed temporary image on the canvas
        try:
             img_bbox = self.canvas.bbox(self.temp_image_display_id)
             if not img_bbox:
                 print("Warning: Could not get bbox for temp image display.")
                 return
             img_cx_min, img_cy_min, img_cx_max, img_cy_max = img_bbox
        except Exception as e:
            print(f"Error getting image bbox: {e}")
            return

        # Calculate click coordinates relative to the TOP-LEFT corner of the displayed image
        click_x_rel = canvas_x - img_cx_min
        click_y_rel = canvas_y - img_cy_min

        # Convert these relative display coordinates back to ORIGINAL image coordinates
        # using the scale factor calculated when displaying the image
        # Check for zero scale factor
        if abs(self.image_display_scale_factor) < 1e-9:
             print("Error: Image display scale factor is zero.")
             return
        original_x = click_x_rel / self.image_display_scale_factor
        original_y = click_y_rel / self.image_display_scale_factor

        # Ensure coordinates are within the original image bounds (sanity check)
        if not self.original_bed_image_pil: # Check if PIL image still exists
             print("Error: Original PIL image not found.")
             self.cancel_corner_selection()
             return
        img_w, img_h = self.original_bed_image_pil.size
        original_x = max(0, min(original_x, img_w - 1))
        original_y = max(0, min(original_y, img_h - 1))

        # Store the ORIGINAL image coordinates
        self.corner_selection_points_temp.append((original_x, original_y))

        # Draw a marker on the canvas at the click position (canvas_x, canvas_y)
        marker_size = 3
        marker_id = self.canvas.create_oval(canvas_x - marker_size, canvas_y - marker_size,
                                            canvas_x + marker_size, canvas_y + marker_size,
                                            fill="yellow", outline="black", tags="corner_marker")
        self.corner_selection_markers.append(marker_id)

        # --- Check if selection is complete ---
        num_selected = len(self.corner_selection_points_temp)
        if num_selected == 4:
            # Selection finished
            if self.corner_selection_mode == 'bed':
                self.bed_image_corners_src = list(self.corner_selection_points_temp) # Store as list
            elif self.corner_selection_mode == 'paper':
                self.paper_image_corners_src = list(self.corner_selection_points_temp) # Store as list

            # Clean up selection mode
            mode_finished = self.corner_selection_mode
            self.corner_selection_mode = None
            if self._corner_click_binding_id:
                self.canvas.unbind("<Button-1>", self._corner_click_binding_id)
                self._corner_click_binding_id = None
            if hasattr(self, '_corner_cancel_binding_id') and self._corner_cancel_binding_id:
                 self.canvas.unbind("<Button-3>", self._corner_cancel_binding_id)
                 self._corner_cancel_binding_id = None

            self.canvas.config(cursor="")
            self.clear_corner_markers() # Clear temp markers
            # Keep the temp image displayed for now, user might want to reselect
            # self.canvas.delete("temp_image_display")
            # self.temp_image_display_id = None

            self.status_label.config(text=f"{mode_finished.capitalize()} corner selection complete.")
            self.update_image_controls_state()

            # If BED corners were just selected, attempt the warp calculation
            if mode_finished == 'bed':
                self.calculate_and_set_warp()
                # Redraw potentially needed if warp fails but state changes
                self.redraw_canvas()
            else:
                # If paper corners selected, maybe update paper overlay?
                # Currently no automatic paper warp feature implemented based on image.
                self.redraw_canvas() # Redraw to remove markers

        else:
            # Update instructions for the next corner
            corner_index = len(self.corner_selection_points_temp)
            self.corner_status_label.config(text=f"Click {CORNER_NAMES[corner_index]} corner for {self.corner_selection_mode.upper()}")
            self.status_label.config(text=f"Corner Selection Mode: Click {self.corner_selection_mode.upper()} {CORNER_NAMES[corner_index]}. Right-click to cancel.")


    def clear_corner_markers(self):
        """Removes visual markers for clicked corners."""
        for marker_id in self.corner_selection_markers:
            self.canvas.delete(marker_id)
        self.corner_selection_markers = []

    def redraw_corner_markers(self):
         """Redraws corner markers if selection is active (needed after canvas clear)."""
         # This requires storing the canvas click coordinates alongside the original image coordinates
         # during selection, or recalculating canvas coordinates from original coords.
         # Recalculating is complex as it depends on the temp image's current display position/scale.
         # For now, markers are cleared on redraw and only reappear as the user clicks again.
         # Let's enhance this by storing canvas coords too.

         # Modify on_corner_click to store:
         # self.corner_selection_points_temp.append({'orig': (original_x, original_y), 'canvas': (canvas_x, canvas_y)})

         # Then, in redraw_corner_markers:
         # self.clear_corner_markers() # Clear old IDs
         # if self.corner_selection_mode and self.corner_selection_points_temp:
         #     marker_size = 3
         #     for point_data in self.corner_selection_points_temp:
         #         cx, cy = point_data['canvas'] # Use stored canvas coords
         #         marker_id = self.canvas.create_oval(cx - marker_size, cy - marker_size,
         #                                             cx + marker_size, cy + marker_size,
         #                                             fill="yellow", outline="black", tags="corner_marker")
         #         self.corner_selection_markers.append(marker_id)

         # **** TEMPORARY FIX: Just clear markers on redraw ****
         self.clear_corner_markers()


    def calculate_and_set_warp(self):
        """Calculates the perspective warp matrix and generates the canonical warped PIL image."""
        if not IMAGE_PROCESSING_ENABLED:
            print("Warp attempt failed: Image processing libraries not available.")
            return
        if self.original_bed_image_cv is None: # Check CV image directly
            self.status_label.config(text="Cannot warp: Original image not loaded.")
            return
        if len(self.bed_image_corners_src) != 4:
            self.status_label.config(text="Cannot warp: Need 4 bed corners selected.")
            return
        # Ensure bed dimensions are valid (update_bed_display_params also sets self.bed_width_mm etc.)
        if not self.update_bed_display_params():
            self.status_label.config(text="Cannot warp: Invalid bed dimensions.")
            return

        try:
            # Source points (from image, clicked order TL, TR, BR, BL)
            # Ensure the points are in the correct order expected by getPerspectiveTransform
            # The user clicks TL, TR, BR, BL. This order should be correct.
            pts_src = np.array(self.bed_image_corners_src, dtype="float32")

            # --- Destination points in canonical pixel space ---
            # Calculate target size based on bed dimensions and desired resolution
            warp_target_w = max(1, int(self.bed_width_mm * self.PIXELS_PER_MM))
            warp_target_h = max(1, int(self.bed_height_mm * self.PIXELS_PER_MM))

            # Destination points corresponding to TL, TR, BR, BL of the bed in the target image space
            # (Using standard image coordinates where Y increases downwards)
            pts_dst = np.array([
                [0, 0],                 # Top-Left (corresponding to user's first click)
                [warp_target_w - 1, 0], # Top-Right (corresponding to user's second click)
                [warp_target_w - 1, warp_target_h - 1], # Bottom-Right (corresponding to user's third click)
                [0, warp_target_h - 1]  # Bottom-Left (corresponding to user's fourth click)
            ], dtype="float32")

            # Calculate the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

            # Check if matrix calculation was successful (it might return None or raise error on degenerate points)
            if matrix is None:
                 raise ValueError("Failed to calculate perspective transform matrix. Check corner points (e.g., are they collinear?).")

            # Warp the original image (stored as OpenCV BGR) to the canonical size
            warped_cv = cv2.warpPerspective(self.original_bed_image_cv, matrix, (warp_target_w, warp_target_h))

            # Convert warped image (BGR) to RGB for PIL
            warped_rgb = cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB)
            # --- Store the canonical warped PIL image ---
            self.warped_bed_image_pil = Image.fromarray(warped_rgb)
            # --- Clear the old Tkinter PhotoImage to force recreation on next draw ---
            self.warped_bed_image_tk = None
            
            # Remove any temporary original image upon successful warping
            if self.temp_image_display_id:
                self.canvas.delete("temp_image_display")
                self.temp_image_display_id = None
                self.temp_original_bed_image_tk = None

            self.status_label.config(text="Image warp calculated successfully.")
            self.update_image_controls_state() # Enable show checkbox
            # Automatically show the image after successful warp
            self.show_warped_image_var.set(True)
            # Redraw handled by caller or subsequent actions usually, but call here for immediate effect.
            self.redraw_canvas() # Redraw to display the new warped image

        except Exception as e:
            # Display the actual error in the messagebox
            error_detail = f"Could not warp image:\n{type(e).__name__}: {e}"
            print(f"Warping Error Traceback: {e}", file=sys.stderr) # Print detailed error to console
            import traceback
            traceback.print_exc() # Print full traceback to console

            messagebox.showerror("Warping Error", error_detail)
            self.status_label.config(text="Error during image warping.")
            self.warped_bed_image_pil = None # Clear result on error
            self.warped_bed_image_tk = None
            self.show_warped_image_var.set(False)
            self.update_image_controls_state()
            # Redraw needed to remove potentially partially drawn/invalid warp state
            self.redraw_canvas()


    def clear_image_data(self, clear_pil_obj=True):
        """Clears all data related to the bed image and warp."""
        if self.corner_selection_mode:
            self.cancel_corner_selection()

        if clear_pil_obj:
            self.original_bed_image_pil = None
        # self.original_bed_image_tk = None # Managed by display_temp / redraw_temp
        self.original_bed_image_cv = None
        self.warped_bed_image_pil = None  # Clear canonical warped PIL
        self.warped_bed_image_tk = None   # Clear current display Tk image
        self.bed_image_corners_src = []
        self.paper_image_corners_src = []
        self.image_display_scale_factor = 1.0
        self.clear_corner_markers()

        # Delete canvas items
        self.canvas.delete("temp_image_display")
        self.temp_image_display_id = None
        self.temp_original_bed_image_tk = None # Clear the tk object holder too
        self.canvas.delete("warped_bed_image")

        # Reset UI state
        self.show_warped_image_var.set(False)
        self.update_image_controls_state() # Update button states etc.
        if clear_pil_obj: # Only update status if fully cleared
            self.status_label.config(text="Image data cleared.")
            self.redraw_canvas() # Redraw without image elements

    def toggle_warped_image_display(self):
        """Toggles the display of the warped bed image and redraws the canvas."""
        # Check if the image actually exists before trying to show it
        if self.show_warped_image_var.get() and not self.warped_bed_image_pil:
             self.show_warped_image_var.set(False) # Untick if no image available
             self.status_label.config(text="No warped image available to show.")
             return # Don't redraw yet
             
        # Remove any temporary original image when toggling warped image
        if self.temp_image_display_id:
            self.canvas.delete("temp_image_display")
            self.temp_image_display_id = None
            self.temp_original_bed_image_tk = None

        # Proceed with redraw
        self.redraw_canvas()

        if self.show_warped_image_var.get():
            self.status_label.config(text="Showing warped image.")
        else:
            self.status_label.config(text="Warped image hidden.")

    # --- Saving ---
    def save_file(self):
        """Saves all VISIBLE files, concatenated, with their individual transforms applied."""
        visible_files_data = [f for f in self.gcode_files if f['visible']]
        if not visible_files_data:
            messagebox.showwarning("No Visible Data", "No visible G-code files to save.")
            return

        # Validate that pen offsets don't result in G-code going outside the bed
        invalid_files = []
        for file_data in visible_files_data:
            # Create temporary transform with pen offsets for validation
            validation_transform = file_data['transform'].copy()
            pen_x = validation_transform['pen_x_offset']
            pen_y = validation_transform['pen_y_offset']
            
            # Get bounds of file with full transform (including pen offsets)
            segments_with_pen = apply_transformations_to_coords(
                file_data['original_segments'], 
                {**validation_transform, 
                 'x_offset': validation_transform['x_offset'] + pen_x,
                 'y_offset': validation_transform['y_offset'] + pen_y}
            )
            
            # Calculate bounds
            if segments_with_pen:
                min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
                for segment in segments_with_pen:
                    for point in segment:
                        min_x = min(min_x, point['x'])
                        max_x = max(max_x, point['x'])
                        min_y = min(min_y, point['y'])
                        max_y = max(max_y, point['y'])
                
                # Check if bounds exceed bed dimensions
                bed_width = self.bed_width_var.get()
                bed_height = self.bed_height_var.get()
                
                if min_x < 0 or min_y < 0 or max_x > bed_width or max_y > bed_height:
                    invalid_files.append({
                        'name': file_data['basename'],
                        'bounds': [min_x, min_y, max_x, max_y],
                        'bed': [bed_width, bed_height]
                    })
        
        # If any files have invalid bounds due to pen offsets, show error
        if invalid_files and not self.skip_pen_warning_var.get():
            error_msg = "Pen offsets would cause these files to go outside the bed:\n\n"
            for file in invalid_files:
                bounds = file['bounds']
                error_msg += f"• {file['name']}: "
                if bounds[0] < 0:
                    error_msg += f"X min ({bounds[0]:.2f}) < 0, "
                if bounds[1] < 0:
                    error_msg += f"Y min ({bounds[1]:.2f}) < 0, "
                if bounds[2] > file['bed'][0]:
                    error_msg += f"X max ({bounds[2]:.2f}) > bed width ({file['bed'][0]:.2f}), "
                if bounds[3] > file['bed'][1]:
                    error_msg += f"Y max ({bounds[3]:.2f}) > bed height ({file['bed'][1]:.2f})"
                error_msg += "\n"
            error_msg += "\nPlease adjust pen offsets or other transforms and try again."
            
            messagebox.showerror("Out of Bounds Error", error_msg)
            self.status_label.config(text="Save failed: G-code would go outside bed due to pen offsets.")
            return

        # Suggest filename based on the first visible file
        try: # Add try-except for safety if file list somehow becomes empty between check and use
             base, ext = os.path.splitext(os.path.basename(visible_files_data[0]['filepath']))
             suggested_filename = f"{base}_merged_modified{ext}"
        except IndexError:
             suggested_filename = "merged_modified.gcode"


        save_path = filedialog.asksaveasfilename(
            title="Save Merged Modified G-Code",
            initialfile=suggested_filename,
            defaultextension=".gcode",
            filetypes=[("G-Code Files", "*.gcode *.nc *.g"), ("All Files", "*.*")]
        )
        if not save_path: return

        try:
            final_gcode_to_save = []
            # Add header
            final_gcode_to_save.append("; G-code merged and modified by Visualizer Tool")
            final_gcode_to_save.append(f"; Merging {len(visible_files_data)} visible file(s)")
            final_gcode_to_save.append(";")
            final_gcode_to_save.append(f"; Bed Dimensions Used: {self.bed_width_var.get():.3f} x {self.bed_height_var.get():.3f} mm")
            final_gcode_to_save.append(";")


            # Process each visible file
            for file_data in visible_files_data:
                filename = file_data['basename']
                transform = file_data['transform']
                final_gcode_to_save.append(f"; --- Start: {filename} ---")
                final_gcode_to_save.append(f"; Transform: Scale(X:{transform['x_scale']:.4f} Y:{transform['y_scale']:.4f} Z:{transform['z_scale']:.4f}) Offset(X:{transform['x_offset']:.4f} Y:{transform['y_offset']:.4f} Z:{transform['z_offset']:.4f}) Pen(X:{transform['pen_x_offset']:.4f} Y:{transform['pen_y_offset']:.4f})")

                # Generate transformed G-code for this specific file
                transformed_lines = generate_transformed_gcode(
                    file_data['original_lines'], transform
                )
                final_gcode_to_save.extend(transformed_lines)
                final_gcode_to_save.append(f"; --- End: {filename} ---")
                final_gcode_to_save.append(";") # Separator

            # Write to file
            with open(save_path, 'w') as f:
                for line in final_gcode_to_save:
                    f.write(line + '\n')

            messagebox.showinfo("Save Successful", f"Merged modified G-code saved to:\n{save_path}")
            self.status_label.config(text=f"Saved merged file to {os.path.basename(save_path)}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file:\n{e}")
            self.status_label.config(text="Save failed.")


    def on_controls_frame_configure(self, event):
        """Update the scrollable region when the inner frame size changes"""
        # Update the canvas's scroll region to encompass the inner frame
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))
        
    def on_controls_canvas_configure(self, event):
        """Resize the inner frame when the canvas is resized"""
        # Adjust the inner frame's width to match the canvas
        self.controls_canvas.itemconfig(self.controls_canvas_window, width=event.width)
        
    def on_mousewheel_controls(self, event):
        """Handle mouse wheel events for scrolling the controls frame"""
        # Only process mousewheel if cursor is over the controls canvas
        x, y = self.winfo_pointerxy()
        widget_under_cursor = self.winfo_containing(x, y)
        
        # Check if widget_under_cursor is the controls_canvas or a child of it
        in_controls = False
        if widget_under_cursor:
            parent = widget_under_cursor
            while parent:
                if parent == self.controls_canvas:
                    in_controls = True
                    break
                try:
                    parent = parent.master
                except AttributeError:
                    break
        
        if not in_controls:
            return
            
        # Calculate scroll amount based on event type
        scroll_amount = 0
        if hasattr(event, 'delta'):  # Windows/macOS
            scroll_amount = -1 * (event.delta // 120) * 30
        elif hasattr(event, 'num'):  # Linux
            if event.num == 4:  # Scroll up
                scroll_amount = -30
            elif event.num == 5:  # Scroll down
                scroll_amount = 30
                
        self.controls_canvas.yview_scroll(scroll_amount, "units")


# --- Main Execution ---
if __name__ == "__main__":
    app = GCodeVisualizer()
    app.mainloop()

# --- END OF FILE gui.py ---
