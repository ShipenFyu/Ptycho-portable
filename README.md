# Ptychography Toolkit

A portable Tkinter-based toolkit for common ptychography visualization workflows.

The app currently includes:

- **Intensity Profile**: load one or more images and plot averaged intensity profiles in axis, line, or multi-image line mode.
- **HSV Fusion**: load amplitude/phase tensors from an `.npz` file, select an ROI, and export an HSV-fused RGB result.

## Requirements

Use Python 3.10 or newer.

Install the runtime dependencies:

```powershell
pip install numpy matplotlib pillow
```

Tkinter is also required. It is included with most standard Python installers on Windows.

## Start The App

Run the toolkit from the project root:

```powershell
python main.py
```

The welcome page opens first. Select a module card to enter the workspace. After entering the workspace, modules are available as notebook tabs.

## Intensity Profile

Use this module to inspect intensity along a vertical/horizontal scan range, a selected line, or per-image lines across multiple images.

Supported image formats:

- `.png`
- `.jpg` / `.jpeg`
- `.bmp`
- `.tif` / `.tiff`

Basic workflow:

1. Click **Add Images (1+)** and select one or more images.
2. Choose **Profile Mode**:
   - **axis**: profile along a fixed width or height position.
   - **line**: click two points on the image to define a line profile.
   - **multi**: define one line per image, useful for comparing different images with different line placements.
3. Adjust parameters:
   - **Fixed Axis**, **Fixed Ratio**, **Scan Start Ratio**, **Scan End Ratio** for axis mode.
   - **Line Samples** for line and multi modes.
   - **Window Size** for averaging thickness. It must be an odd integer.
4. Click **Plot Profile** if the profile does not update automatically.
5. Click **Save Plot** to export the result.

Notes:

- In **axis** and **line** mode, images must have the same height/width ratio when loaded together.
- In **multi** mode, each image can have its own line. The profile x-axis and intensity are normalized for comparison.
- **Save Mode** controls whether the full image/profile composite or only the profile chart is saved.

## HSV Fusion

Use this module to combine amplitude and phase tensors into an RGB visualization.

Input format:

- One `.npz` file containing real-valued 2D arrays.
- The selected amplitude and phase tensors must have the same shape.

Basic workflow:

1. Click **Load NPZ + Select Tensors**.
2. Choose the amplitude tensor key and phase tensor key in the popup.
3. Use **Prev Input** / **Next Input** to switch the preview between amplitude and phase.
4. Optionally enable **Square ROI**.
5. Drag on the input preview to select an ROI.
6. Adjust **Saturation [0, 1]** if needed. Press Enter or leave the field to refresh the output.
7. Click **Save HSV Result** to export the fused image.

Phase handling:

- If phase values are within `[-pi, pi]`, hue is mapped across that range.
- Otherwise, hue is normalized across the selected phase ROI value range.

## Project Layout

Feature modules live under `modules/`:

```text
modules/
  base_module.py
  intensity_profile/
  hsv_fusion/
  welcome/
```

Each feature package exposes its public class through `__init__.py`. The usual pattern is:

- `module.py` or `screen.py`: Tkinter UI, event handlers, and orchestration.
- `models.py`: shared data classes.
- loader/adapter/processor files: pure loading, validation, and computation logic.

This keeps UI code separate from logic that can be tested independently.

