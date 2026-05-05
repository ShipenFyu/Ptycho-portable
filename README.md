# Ptychography Toolkit

A portable Tkinter-based toolkit for common ptychography visualization workflows.

The app currently includes:

- **Intensity Profile**: load one or more images and plot averaged intensity profiles in axis, line, or multi-image line mode.
- **HSV Fusion**: load amplitude/phase tensors from an `.npz` file, select an ROI, and export an HSV-fused RGB result.
- **Resolution**: estimate image resolution. The current method is knife-edge analysis using ESF, LSF, and MTF curves.

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

## Resolution

Use this module to estimate image resolution. The current implementation uses the knife-edge method on a single image ROI.

Supported image formats:

- `.png`
- `.jpg` / `.jpeg`
- `.bmp`
- `.tif` / `.tiff`

Basic workflow:

1. Click **Load Image**.
2. Drag a ROI around one clean knife edge on the left image.
3. Click **Analyze ROI**.
4. Check the cyan detected edge line on the original image and ROI preview.
5. Read **MTF50**, **MTF10**, and **10-90 width** in the result panel.
6. Optionally enter a real **Pixel Size** and unit to convert frequency to `lp/mm` and edge width to physical units.
7. Click **Save Plot** or **Save CSV** to export the curves and measurements.

Use **Zoom In**, **Zoom Out**, **Reset**, or the mouse wheel over the original image to enlarge the view before drawing the ROI.

ROI selection tips:

- The ROI should contain one edge only.
- The edge should pass through most of the ROI.
- Keep both bright and dark flat regions inside the ROI.
- Avoid corners, texture, strong artifacts, or multiple transitions.
- A slightly slanted edge is preferred because it improves oversampled ESF reconstruction.

Measured curves:

- **ESF**: edge spread function, sampled across the detected edge.
- **LSF**: derivative of the smoothed ESF.
- **MTF**: normalized Fourier magnitude of the LSF.

Without pixel size, frequency is reported in `cycles/pixel`. With pixel size, it is also reported in `lp/mm`.

## Project Layout

Feature modules live under `modules/`:

```text
modules/
  base_module.py
  intensity_profile/
  hsv_fusion/
  resolution/
  welcome/
```

Each feature package exposes its public class through `__init__.py`. The usual pattern is:

- `module.py` or `screen.py`: Tkinter UI, event handlers, and orchestration.
- `models.py`: shared data classes.
- loader/adapter/processor files: pure loading, validation, and computation logic.

This keeps UI code separate from logic that can be tested independently.
