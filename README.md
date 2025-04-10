# CodePlotter

A Python-based tool for visualizing, manipulating, and merging G-code files. This application is particularly useful for pen plotting with 3D printers like the Ender 3 V2. It provides a graphical interface for editing G-code files, applying transformations, and visualizing the toolpath on a virtual build plate.

---

## Features

- **Multi-file support**: Load and manage multiple G-code files simultaneously.
- **Visual preview**: See your G-code paths visualized on a virtual build plate.
- **Transformation tools**:
  - Scale, offset, and rotate G-code paths.
  - Apply pen offsets for pen plotter mounts.
- **Paper overlay**:
  - Add overlays for A4, US Letter, or College Ruled paper.
  - Adjust paper position with offsets.
- **File merging**: Combine multiple G-code files into a single output file.
- **Interactive manipulation**:
  - Drag and drop files directly on the canvas.
  - Center designs or apply margins with a single click.
- **Export**: Save merged and transformed G-code files ready for printing.

---

## Use Cases

This tool is ideal for:
1. Converting 3D printer movements to pen plotting.
2. Arranging multiple designs on a single sheet of paper.
3. Visualizing G-code before sending it to your printer.
4. Adjusting for hardware offsets, such as pen mounts.
5. Merging multiple files for continuous printing.

---

## Installation

### Prerequisites
- Python 3.6 or higher.
- Tkinter (usually bundled with Python).

### Setup
1. Clone this repository or download the source files.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Application
To start the graphical interface:
```bash
python gui.py
```

To use the command-line version:
```bash
python legacy.py
```

### Basic Workflow

1. **Load G-code file(s)**:
   - Click the "Load" button to import one or more G-code files.
   
2. **Visualize and transform**:
   - Select a file from the list to modify its properties.
   - Adjust scale, offsets, and pen offsets as needed.
   - Drag files directly on the canvas for positioning.
   
3. **Add paper overlay**:
   - Select paper type and position for planning.
   - Offset the paper as needed to match your printer setup.
   
4. **Save merged results**:
   - Click "Save Visible Merged" to export a combined G-code file with all transformations applied.

---

## Pen Plotter Setup

For using an Ender 3 V2 as a pen plotter:

1. Set appropriate pen offsets to compensate for the mount position.
2. Use the paper overlay to visualize the workspace boundaries.
3. Apply margin offsets to ensure the pen stays within the paper.
4. Adjust Z offsets as needed for pen contact/release.

---

## Key Features in Detail

### Paper Overlay
- Supported paper types:
  - A4 (210mm x 297mm)
  - US Letter (215.9mm x 279.4mm)
  - College Ruled (203.2mm x 266.7mm)
- Add an overlay to the virtual build plate to plan your design placement.

### Transformations
- **Scale**: Adjust the size of the G-code paths.
- **Offset**: Move the paths to a specific position on the build plate.
- **Pen Offset**: Compensate for the position of the pen mount on your printer.

### File Merging
- Combine multiple G-code files into a single output file.
- Each file retains its individual transformations.

### Visualization
- View the toolpath for each file in different colors.
- Toggle visibility for individual files.
- See the combined bounds of all visible files.

---

## Building an Executable

To create a standalone executable file that doesn't require Python installation:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Build the executable:
   ```bash
   pyinstaller --onefile --windowed gui.py
   ```

3. The executable will be created in the `dist` folder.

---

## Project Structure

- `gui.py`: Main application with a graphical user interface.
- `legacy.py`: Command-line version for basic G-code manipulation.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation for the project.

---

## Tips and Tricks

- **Drag and Drop**: Click and drag G-code paths directly on the canvas for intuitive positioning.
- **Center on Bed**: Use the "Center Selected on Bed" button to quickly center a design.
- **Safe Margins**: Apply a 10mm margin from the edge with the "Set Selected Margin" button.
- **Visibility Toggle**: Hide/show individual files to focus on specific parts.
- **Custom Colors**: Change the color of each file for better visualization.
- **Z-movement Filtering**: Hide Z-only movements for cleaner visualization.

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for feature requests or bug reports.

---

## License

This project is open source under the MIT license.