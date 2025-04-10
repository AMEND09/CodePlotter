# --- START OF FILE gui.py ---

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import re
import math
import os
import itertools # For color cycling

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


# --- Core G-code processing functions ---

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
    pen_x_offset = transform_params['pen_x_offset']
    pen_y_offset = transform_params['pen_y_offset']

    for segment in original_coords_segments:
        new_segment = []
        for point in segment:
            # Apply scale and primary offset (to original relative coords)
            transformed_x = (point['x'] * x_scale) + x_offset
            transformed_y = (point['y'] * y_scale) + y_offset
            # Apply pen offset
            final_x = transformed_x + pen_x_offset
            final_y = transformed_y + pen_y_offset
            # Copy other info
            new_segment.append({**point, 'x': final_x, 'y': final_y})
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

    for line in original_gcode_lines:
        modified_line = line
        parts = line.split(';', 1)
        code_part = parts[0]
        comment_part = f";{parts[1]}" if len(parts) > 1 else ""

        cleaned_code_upper = code_part.strip().upper()
        is_motion_cmd = cleaned_code_upper.startswith('G0') or cleaned_code_upper.startswith('G1')
        is_set_coord_cmd = cleaned_code_upper.startswith('G92')

        def replace_coord(match):
            axis = match.group(1).upper()
            try:
                val = float(match.group(2))
                new_val = val

                if axis == 'X':
                    new_val = (val * x_scale) + x_offset
                    if is_motion_cmd or is_set_coord_cmd: new_val += pen_x_offset
                    return f'X{new_val:.6f}'
                elif axis == 'Y':
                    new_val = (val * y_scale) + y_offset
                    if is_motion_cmd or is_set_coord_cmd: new_val += pen_y_offset
                    return f'Y{new_val:.6f}'
                elif axis == 'Z':
                    if is_motion_cmd or is_set_coord_cmd:
                        new_val = (val * z_scale) + z_offset
                        return f'Z{new_val:.6f}'
                    else: return match.group(0)
                else: return match.group(0)
            except ValueError: return match.group(0)

        modified_code_part = coord_pattern.sub(replace_coord, code_part)
        modified_gcode.append(modified_code_part + comment_part)

    return modified_gcode


# --- GUI Application Class ---

class GCodeVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("G-Code Visualizer & Manipulator (Multi-File)")
        self.geometry("1150x800") # Wider for file list

        # --- Data Storage ---
        self.gcode_files = [] # List of dictionaries, one per file
        self.selected_file_index = tk.IntVar(value=-1) # Track listbox selection index

        # --- Transformation State Variables (reflect selected file) ---
        self.current_x_scale = tk.DoubleVar(value=1.0)
        self.current_y_scale = tk.DoubleVar(value=1.0)
        self.current_z_scale = tk.DoubleVar(value=1.0)
        self.current_x_offset = tk.DoubleVar(value=0.0)
        self.current_y_offset = tk.DoubleVar(value=0.0)
        self.current_z_offset = tk.DoubleVar(value=0.0)
        self.pen_offset_x_var = tk.DoubleVar(value=0.0)
        self.pen_offset_y_var = tk.DoubleVar(value=0.0)
        self.current_visibility_var = tk.BooleanVar(value=True) # Reflects selected file's visibility

        # --- Bed Settings State ---
        self.bed_width_var = tk.DoubleVar(value=220.0)
        self.bed_height_var = tk.DoubleVar(value=220.0)

        # --- Paper Overlay State ---
        self.paper_type_var = tk.StringVar(value="None")
        self.paper_offset_x_var = tk.DoubleVar(value=0.0)
        self.paper_offset_y_var = tk.DoubleVar(value=0.0)

        # --- View State ---
        self.hide_z_moves_var = tk.BooleanVar(value=False)

        # --- Canvas Setup ---
        self.canvas_view_width = 650 # Adjusted canvas size
        self.canvas_view_height = 650
        self.padding = 30

        # --- UI Elements ---
        self.create_widgets()

        # --- Mouse Bindings ---
        self.canvas.tag_bind("gcode_path", "<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_stop)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<Alt-ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<Alt-ButtonRelease-1>", self.on_pan_stop)
        self.canvas.bind("<Alt-B1-Motion>", self.on_pan_motion)

        # --- Drag/Pan State Variables ---
        self._pan_data = {"x": 0, "y": 0}
        self._drag_data = {
            "start_offset_x": 0, "start_offset_y": 0,
            "last_event_x": 0, "last_event_y": 0,
            "is_dragging": False,
            "dragged_file_index": -1 # Index of the file being dragged
        }

        # --- Initialize ---
        self.update_bed_display_params()
        self.draw_bed()
        self.update_ui_for_selection() # Disable controls initially

    # --- Coordinate Conversion (Unchanged) ---
    def update_bed_display_params(self):
        """Recalculates scaling and corner pixels based on bed size vars and canvas view size."""
        try:
            self.bed_width_mm = self.bed_width_var.get()
            self.bed_height_mm = self.bed_height_var.get()
            if self.bed_width_mm <= 0 or self.bed_height_mm <= 0:
                raise ValueError("Bed dimensions must be positive.")

            available_width = self.canvas_view_width - 2 * self.padding
            available_height = self.canvas_view_height - 2 * self.padding
            if available_width <=0 or available_height <=0: available_width=available_height=1

            scale_x = available_width / self.bed_width_mm
            scale_y = available_height / self.bed_height_mm
            self.canvas_scale = min(scale_x, scale_y) # Maintain aspect ratio

            bed_pixel_width = self.bed_width_mm * self.canvas_scale
            bed_pixel_height = self.bed_height_mm * self.canvas_scale

            self.bed_cx_min = self.padding + (available_width - bed_pixel_width) / 2
            self.bed_cy_max = self.canvas_view_height - self.padding - (available_height - bed_pixel_height) / 2 # Y=0 is bottom
            self.bed_cx_max = self.bed_cx_min + bed_pixel_width
            self.bed_cy_min = self.bed_cy_max - bed_pixel_height

            return True
        except Exception as e:
            messagebox.showerror("Bed Size Error", f"Invalid bed dimensions: {e}")
            self.bed_width_var.set(220.0)
            self.bed_height_var.set(220.0)
            self.update_bed_display_params() # Recalculate with defaults
            return False

    def update_bed_action(self):
        """Action for the Update Bed button."""
        if self.update_bed_display_params():
            self.draw_bed()
            self.redraw_canvas()
            self.status_label.config(text="Bed dimensions updated.")

    def gcode_to_canvas(self, x, y):
        """Converts G-code coordinates (mm, origin bottom-left) to canvas pixel coordinates."""
        cx = self.bed_cx_min + (x * self.canvas_scale)
        cy = self.bed_cy_max - (y * self.canvas_scale) # Flip Y axis
        return cx, cy

    def canvas_to_gcode(self, cx, cy):
        """Converts canvas pixel coordinates back to G-code coordinates (mm)."""
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
        left_frame.pack_propagate(False) # Prevent resizing

        controls_frame = ttk.Frame(self) # Frame for transform controls
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left Frame Widgets (File List & Management) ---
        file_list_frame = ttk.LabelFrame(left_frame, text="Loaded Files", padding=5)
        file_list_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(file_list_frame, exportselection=False) # Keep selection on focus loss
        self.file_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        file_buttons_frame = ttk.Frame(file_list_frame)
        file_buttons_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))

        self.load_button = ttk.Button(file_buttons_frame, text="Load", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.remove_button = ttk.Button(file_buttons_frame, text="Remove", command=self.remove_selected_file, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.clear_all_button = ttk.Button(file_buttons_frame, text="Clear All", command=self.reset_app_state, state=tk.DISABLED)
        self.clear_all_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # --- Controls Frame Widgets (Transformations for Selected File) ---
        controls_frame.columnconfigure(0, weight=1) # Make column expandable

        row_idx = 0
        # --- File Operations (Moved Save here) ---
        file_op_frame = ttk.LabelFrame(controls_frame, text="File Operations", padding=5)
        file_op_frame.grid(row=row_idx, column=0, pady=(0,5), sticky="ew"); row_idx += 1
        self.save_button = ttk.Button(file_op_frame, text="Save Visible Merged", command=self.save_file, state=tk.DISABLED)
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
        selected_file_frame.columnconfigure(2, weight=1) # Push reset button right


        # --- Bed Settings ---
        bed_frame = ttk.LabelFrame(controls_frame, text="Bed Settings (mm)", padding="5")
        bed_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        # (Bed widgets unchanged)
        ttk.Label(bed_frame, text="Width:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.bed_width_entry = ttk.Entry(bed_frame, textvariable=self.bed_width_var, width=7)
        self.bed_width_entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(bed_frame, text="Height:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.bed_height_entry = ttk.Entry(bed_frame, textvariable=self.bed_height_var, width=7)
        self.bed_height_entry.grid(row=1, column=1, padx=2, pady=2)
        self.update_bed_button = ttk.Button(bed_frame, text="Update Bed", command=self.update_bed_action)
        self.update_bed_button.grid(row=0, column=2, rowspan=2, padx=10, pady=2, sticky="ns")


        # --- Paper Overlay ---
        paper_frame = ttk.LabelFrame(controls_frame, text="Paper Overlay", padding="5")
        paper_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        # (Paper widgets unchanged)
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

        # --- View Options ---
        view_frame = ttk.LabelFrame(controls_frame, text="Global View Options", padding="5")
        view_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        self.hide_z_check = ttk.Checkbutton(view_frame, text="Hide Z-only moves (All Files)", variable=self.hide_z_moves_var, command=self.redraw_canvas)
        self.hide_z_check.pack(anchor="w")

        # --- Transformation Entries (Apply to Selected) ---
        transform_frame = ttk.LabelFrame(controls_frame, text="Transform Selected File", padding=5)
        transform_frame.grid(row=row_idx, column=0, pady=5, sticky="ew"); row_idx += 1
        transform_frame.columnconfigure(1, weight=1) # Allow entries to expand slightly if needed

        # Scale
        ttk.Label(transform_frame, text="Scale X:").grid(row=0, column=0, padx=2, pady=1, sticky="w")
        self.scale_x_entry = ttk.Entry(transform_frame, textvariable=self.current_x_scale, width=10, state=tk.DISABLED)
        self.scale_x_entry.grid(row=0, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Scale Y:").grid(row=1, column=0, padx=2, pady=1, sticky="w")
        self.scale_y_entry = ttk.Entry(transform_frame, textvariable=self.current_y_scale, width=10, state=tk.DISABLED)
        self.scale_y_entry.grid(row=1, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Scale Z:").grid(row=2, column=0, padx=2, pady=1, sticky="w")
        self.scale_z_entry = ttk.Entry(transform_frame, textvariable=self.current_z_scale, width=10, state=tk.DISABLED)
        self.scale_z_entry.grid(row=2, column=1, padx=2, pady=1, sticky="ew")

        # Offset
        ttk.Label(transform_frame, text="Offset X:").grid(row=3, column=0, padx=2, pady=1, sticky="w")
        self.offset_x_entry = ttk.Entry(transform_frame, textvariable=self.current_x_offset, width=10, state=tk.DISABLED)
        self.offset_x_entry.grid(row=3, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Offset Y:").grid(row=4, column=0, padx=2, pady=1, sticky="w")
        self.offset_y_entry = ttk.Entry(transform_frame, textvariable=self.current_y_offset, width=10, state=tk.DISABLED)
        self.offset_y_entry.grid(row=4, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Offset Z:").grid(row=5, column=0, padx=2, pady=1, sticky="w")
        self.offset_z_entry = ttk.Entry(transform_frame, textvariable=self.current_z_offset, width=10, state=tk.DISABLED)
        self.offset_z_entry.grid(row=5, column=1, padx=2, pady=1, sticky="ew")

        # Pen Offset
        ttk.Label(transform_frame, text="Pen Offset X:").grid(row=6, column=0, padx=2, pady=1, sticky="w")
        self.pen_offset_x_entry = ttk.Entry(transform_frame, textvariable=self.pen_offset_x_var, width=10, state=tk.DISABLED)
        self.pen_offset_x_entry.grid(row=6, column=1, padx=2, pady=1, sticky="ew")
        ttk.Label(transform_frame, text="Pen Offset Y:").grid(row=7, column=0, padx=2, pady=1, sticky="w")
        self.pen_offset_y_entry = ttk.Entry(transform_frame, textvariable=self.pen_offset_y_var, width=10, state=tk.DISABLED)
        self.pen_offset_y_entry.grid(row=7, column=1, padx=2, pady=1, sticky="ew")

        # Apply Button for selected file's transforms
        self.apply_transforms_button = ttk.Button(transform_frame, text="Apply to Selected", command=self.apply_transformations_to_selected, state=tk.DISABLED)
        self.apply_transforms_button.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # General Actions (Center, Margin - act on selected file?)
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
        self.status_label = ttk.Label(status_frame, text="", wraplength=280, justify=tk.LEFT)
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

    def load_file(self):
        filepaths = filedialog.askopenfilenames(
            title="Open G-Code File(s)",
            filetypes=[("G-Code Files", "*.gcode *.nc *.g"), ("All Files", "*.*")]
        )
        if not filepaths: return

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
        self.canvas.delete("gcode_path", "bounds_rect", "paper_overlay")
        self.update_bed_display_params()
        self.draw_bed()
        self.canvas.configure(scrollregion=self.canvas.bbox("bed"))
        self.update_status_files_loaded()
        self.bounds_label.config(text="Visible Bounds: N/A")
        self.status_label.config(text="Cleared all files.")

    def update_status_files_loaded(self):
         self.files_loaded_label.config(text=f"Files loaded: {len(self.gcode_files)}")

    # --- Transformations for Selected File ---

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
            if not self.update_bed_display_params(): return

        self.canvas.create_rectangle(
            self.bed_cx_min, self.bed_cy_min, self.bed_cx_max, self.bed_cy_max,
            outline="darkgray", width=2, tags=("bed", "bed_visual")
        )
        grid_spacing_mm = 10
        num_x = max(0, int(self.bed_width_mm // grid_spacing_mm))
        num_y = max(0, int(self.bed_height_mm // grid_spacing_mm))
        for i in range(1, num_x + 1):
            cx, _ = self.gcode_to_canvas(i * grid_spacing_mm, 0)
            self.canvas.create_line(cx, self.bed_cy_min, cx, self.bed_cy_max, fill="lightgray", tags=("bed", "bed_visual"))
        for i in range(1, num_y + 1):
            _, cy = self.gcode_to_canvas(0, i * grid_spacing_mm)
            self.canvas.create_line(self.bed_cx_min, cy, self.bed_cx_max, cy, fill="lightgray", tags=("bed", "bed_visual"))
        cx0, cy0 = self.gcode_to_canvas(0, 0)
        self.canvas.create_oval(cx0-3, cy0-3, cx0+3, cy0+3, fill="red", outline="red", tags=("bed", "bed_visual"))
        self.canvas.create_text(cx0 + 5, cy0 + 5, text="(0,0)", anchor="nw", fill="red", tags=("bed", "bed_visual"), font=("Arial", 8))
        self.canvas.tag_lower("bed_visual")

    def draw_paper_overlay(self):
        """Draws the selected paper size at the specified offset."""
        self.canvas.delete("paper_overlay")
        paper_type = self.paper_type_var.get()
        if paper_type == "None" or paper_type not in PAPER_SIZES:
            return
        try:
            offset_x = self.paper_offset_x_var.get()
            offset_y = self.paper_offset_y_var.get()
            width, height = PAPER_SIZES[paper_type]
            g_min_x, g_min_y = offset_x, offset_y
            g_max_x, g_max_y = offset_x + width, offset_y + height
            c_min_x, c_max_y = self.gcode_to_canvas(g_min_x, g_min_y)
            c_max_x, c_min_y = self.gcode_to_canvas(g_max_x, g_max_y)
            self.canvas.create_rectangle(
                c_min_x, c_min_y, c_max_x, c_max_y,
                outline="purple", dash=(4, 4), width=1, tags="paper_overlay"
            )
            self.canvas.tag_raise("paper_overlay", "bed_visual")
            self.canvas.tag_lower("paper_overlay", "gcode_path") # Ensure paper is behind gcode
            self.canvas.tag_lower("paper_overlay", "bounds_rect")
        except ValueError: self.status_label.config(text="Invalid paper offset value.")
        except Exception as e: self.status_label.config(text=f"Error drawing paper: {e}")

    def on_paper_type_change(self, event=None):
        self.draw_paper_overlay()
        self.update_scroll_region()

    def update_paper_overlay_action(self):
        self.draw_paper_overlay()
        self.update_scroll_region()
        self.status_label.config(text="Paper overlay updated.")


    def draw_gcode(self):
        """Draws G-code paths for all VISIBLE files using their respective colors and current segments."""
        self.canvas.delete("gcode_path", "bounds_rect") # Clear previous paths/bounds
        if not self.gcode_files: return

        hide_z = self.hide_z_moves_var.get()
        bounds_calculated = False

        # Iterate through each loaded file
        for index, file_data in enumerate(self.gcode_files):
            if not file_data['visible']: continue # Skip hidden files

            file_color = file_data['color']
            file_tag = f"gcode_file_{index}" # Unique tag for this file's path elements

            for segment in file_data['current_segments']: # Use pre-transformed segments
                if len(segment) < 2: continue

                for i in range(1, len(segment)):
                    p1 = segment[i-1]
                    p2 = segment[i]

                    skip_this_line = False
                    if hide_z:
                        if abs(p1['x'] - p2['x']) < GCODE_TOLERANCE and \
                           abs(p1['y'] - p2['y']) < GCODE_TOLERANCE:
                            skip_this_line = True

                    if not skip_this_line:
                        cx1, cy1 = self.gcode_to_canvas(p1['x'], p1['y'])
                        cx2, cy2 = self.gcode_to_canvas(p2['x'], p2['y'])

                        is_travel = p2['cmd'].startswith("G0")
                        # Use file's color, maybe slightly different for travel?
                        line_color = "gray" if is_travel else file_color # Example: Gray for G0, file color for G1
                        line_width = 1 if is_travel else 2

                        # Add general tag AND file-specific tag
                        self.canvas.create_line(cx1, cy1, cx2, cy2, fill=line_color, width=line_width, tags=("gcode_path", file_tag))

        # Draw bounding box for ALL VISIBLE files combined
        combined_bounds = self.get_current_bounds() # Calculates based on visible files
        if combined_bounds:
            cxmin_g, cymin_g = combined_bounds['min_x'], combined_bounds['min_y']
            cxmax_g, cymax_g = combined_bounds['max_x'], combined_bounds['max_y']
            cxmin, c_y_for_min_x = self.gcode_to_canvas(cxmin_g, cymax_g)
            cxmax, c_y_for_max_x = self.gcode_to_canvas(cxmax_g, cymin_g)
            cxmin_final, cymin_final = min(cxmin, cxmax), min(c_y_for_min_x, c_y_for_max_x)
            cxmax_final, cymax_final = max(cxmin, cxmax), max(c_y_for_min_x, c_y_for_max_x)
            self.canvas.create_rectangle(cxmin_final, cymin_final, cxmax_final, cymax_final,
                                         outline="green", dash=(2, 2), tags="bounds_rect")
            bounds_calculated = True

        # Update bounds label based on combined bounds
        if bounds_calculated:
             info = (f"Visible Bounds: X[{combined_bounds['min_x']:.2f}..{combined_bounds['max_x']:.2f}] "
                     f"Y[{combined_bounds['min_y']:.2f}..{combined_bounds['max_y']:.2f}]")
             self.bounds_label.config(text=info)
        else:
             self.bounds_label.config(text="Visible Bounds: N/A (No visible paths)")

        # Ensure drawing order (paths/bounds above paper/bed)
        self.canvas.tag_raise("gcode_path")
        self.canvas.tag_raise("bounds_rect")


    def redraw_canvas(self):
        """Redraws everything: Bed, Paper, G-code for visible files, updates scroll."""
        self.canvas.delete("all") # Clear everything before full redraw
        self.draw_bed()
        self.draw_paper_overlay()
        self.draw_gcode() # Draws visible files and combined bounds rect
        self.update_scroll_region()


    def update_scroll_region(self):
         """Updates the canvas scrollregion to fit all drawn items."""
         try:
             bbox = self.canvas.bbox("all")
             if bbox:
                 padded_bbox = (bbox[0]-50, bbox[1]-50, bbox[2]+50, bbox[3]+50)
                 self.canvas.configure(scrollregion=padded_bbox)
             else:
                 bed_bbox = self.canvas.bbox("bed")
                 if bed_bbox:
                      padded_bbox = (bed_bbox[0]-50, bed_bbox[1]-50, bed_bbox[2]+50, bed_bbox[3]+50)
                      self.canvas.configure(scrollregion=padded_bbox)
                 else:
                      self.canvas.configure(scrollregion=(0,0,self.canvas_view_width, self.canvas_view_height))
         except Exception as e:
              print(f"Error updating scroll region: {e}")
              self.canvas.configure(scrollregion=(0,0,self.canvas_view_width, self.canvas_view_height))

    # --- Bounds Calculation ---

    def get_bounds_for_file(self, index):
        """Calculates bounds for a specific file index based on its current_segments."""
        if not (0 <= index < len(self.gcode_files)): return None
        file_data = self.gcode_files[index]
        if not file_data['current_segments']: return None

        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        has_points = False
        for segment in file_data['current_segments']:
            for point in segment:
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
                     has_any_points = True
                     overall_min_x = min(overall_min_x, point['x'])
                     overall_max_x = max(overall_max_x, point['x'])
                     overall_min_y = min(overall_min_y, point['y'])
                     overall_max_y = max(overall_max_y, point['y'])

        if has_any_points:
            return {"min_x": overall_min_x, "max_x": overall_max_x, "min_y": overall_min_y, "max_y": overall_max_y}
        else:
            return None


    # --- Mouse Handlers (Panning unchanged, Dragging adapted) ---

    def on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self._pan_data["x"] = event.x; self._pan_data["y"] = event.y
        self.canvas.config(cursor="fleur")

    def on_pan_motion(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_pan_stop(self, event):
        self.canvas.config(cursor="")


    def on_drag_start(self, event):
        # Find the topmost item under cursor with a file-specific tag
        items = self.canvas.find_overlapping(event.x-1, event.y-1, event.x+1, event.y+1)
        dragged_file_index = -1
        for item in reversed(items): # Check topmost first
            tags = self.canvas.gettags(item)
            if "gcode_path" in tags:
                for tag in tags:
                    if tag.startswith("gcode_file_"):
                        try:
                            dragged_file_index = int(tag.split("_")[-1])
                            break # Found file index
                        except ValueError:
                            continue
                if dragged_file_index != -1:
                    break # Stop searching items

        if 0 <= dragged_file_index < len(self.gcode_files):
            # Select the dragged file in the listbox
            if self.selected_file_index.get() != dragged_file_index:
                 self.file_listbox.selection_clear(0, tk.END)
                 self.file_listbox.selection_set(dragged_file_index)
                 self.file_listbox.see(dragged_file_index)
                 # self.selected_file_index.set(dragged_file_index) # on_file_select will handle this
                 self.on_file_select() # Update UI controls

            # Now start the drag state for this specific file
            file_data = self.gcode_files[dragged_file_index]
            self._drag_data["is_dragging"] = True
            self._drag_data["dragged_file_index"] = dragged_file_index
            self._drag_data["last_event_x"] = event.x
            self._drag_data["last_event_y"] = event.y
            # Store the starting primary offset of the *dragged* file
            self._drag_data["start_offset_x"] = file_data['transform']['x_offset']
            self._drag_data["start_offset_y"] = file_data['transform']['y_offset']
            self.canvas.config(cursor="hand2")
            self.status_label.config(text=f"Dragging: {file_data['basename']}")
        else:
            self._drag_data["is_dragging"] = False
            self._drag_data["dragged_file_index"] = -1


    def on_drag_motion(self, event):
        if not self._drag_data["is_dragging"]: return

        index = self._drag_data["dragged_file_index"]
        if not (0 <= index < len(self.gcode_files)):
            self._drag_data["is_dragging"] = False # Should not happen
            return

        # Calculate delta in window pixels
        delta_cx = event.x - self._drag_data["last_event_x"]
        delta_cy = event.y - self._drag_data["last_event_y"]

        # Convert pixel delta to G-code offset delta
        current_canvas_scale = self.canvas_scale
        if abs(current_canvas_scale) < 1e-9: return # Avoid division by zero
        delta_gx = delta_cx / current_canvas_scale
        delta_gy = -delta_cy / current_canvas_scale # Y inverted

        # Calculate the new *total* primary offset for the dragged file
        new_offset_x = self._drag_data["start_offset_x"] + delta_gx
        new_offset_y = self._drag_data["start_offset_y"] + delta_gy

        # --- Update Dragged File's State ---
        # 1. Update the linked Tkinter Variables (immediately updates entry fields)
        self.current_x_offset.set(round(new_offset_x, 6))
        self.current_y_offset.set(round(new_offset_y, 6))

        # 2. Update the file's transform dictionary directly (for immediate segment recalc)
        file_data = self.gcode_files[index]
        file_data['transform']['x_offset'] = new_offset_x
        file_data['transform']['y_offset'] = new_offset_y

        # 3. Recalculate segments for the dragged file
        file_data['current_segments'] = apply_transformations_to_coords(
            file_data['original_segments'], file_data['transform']
        )

        # --- Redraw for Preview ---
        # Option 1: Redraw everything (simplest, might be slow)
        # self.redraw_canvas()
        # Option 2: Only redraw the dragged file (more complex canvas item management)
        # For now, let's stick to redraw_canvas and see performance. If slow, optimise later.
        self.redraw_canvas() # TODO: Optimise if laggy

        # Update last event coords and effective starting offset for next delta calculation
        self._drag_data["last_event_x"] = event.x
        self._drag_data["last_event_y"] = event.y
        self._drag_data["start_offset_x"] = new_offset_x # Update effective start for next move
        self._drag_data["start_offset_y"] = new_offset_y

        # Update status
        self.status_label.config(text=f"Dragging {file_data['basename']}... Offset X:{new_offset_x:.2f} Y:{new_offset_y:.2f}")


    def on_drag_stop(self, event):
        if self._drag_data["is_dragging"]:
            index = self._drag_data["dragged_file_index"]
            self.canvas.config(cursor="")
            self._drag_data["is_dragging"] = False
            self._drag_data["dragged_file_index"] = -1

            # Final state is already set by the last on_drag_motion update
            # and redraw_canvas was called there. Just update status.
            if 0 <= index < len(self.gcode_files):
                 self.status_label.config(text=f"Drag complete for: {self.gcode_files[index]['basename']}")
            else:
                 self.status_label.config(text="Drag complete.")
        # Ensure reset even if something went wrong
        self._drag_data["is_dragging"] = False
        self._drag_data["dragged_file_index"] = -1

    # --- Saving ---

    def save_file(self):
        """Saves all VISIBLE files, concatenated, with their individual transforms applied."""
        visible_files_data = [f for f in self.gcode_files if f['visible']]
        if not visible_files_data:
            messagebox.showwarning("No Visible Data", "No visible G-code files to save.")
            return

        # Suggest filename based on the first visible file
        base, ext = os.path.splitext(os.path.basename(visible_files_data[0]['filepath']))
        suggested_filename = f"{base}_merged_modified{ext}"

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


# --- Main Execution ---
if __name__ == "__main__":
    app = GCodeVisualizer()
    app.mainloop()