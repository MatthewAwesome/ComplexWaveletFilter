# ComplexWaveletFilter_interactive.py
# Save this file next to ComplexWaveletFilter.py and open it in an editor that supports "Run Cell" (VS Code, Spyder).
# Each section is marked with `# %%` so you can execute the script block-by-block while tuning parameters.
# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage import exposure

from ComplexWaveletFilter import (
    calculate_g_and_s,
    process_files,
    process_unfil_files,
    plot_combined_data,
    calculate_and_plot_lifetime
)

# %%
# Pyramid inspection + visualization helpers

def print_pyramid_summary(pyramid, name="pyramid"):
    """
    Print level/band shapes and simple stats for a dtcwt highpasses tuple/list.
    pyramid: sequence of arrays, each array shape (H, W, bands)
    """
    print(f"Summary for {name}: {len(pyramid)} levels (finest -> coarsest)")
    for lvl, arr in enumerate(pyramid):
        if arr is None:
            print(f" level {lvl}: None")
            continue
        h, w, b = arr.shape
        magn = np.abs(arr)
        print(f" level {lvl}: shape={arr.shape}, mag mean={magn.mean():.3e}, mag std={magn.std():.3e}, mag max={magn.max():.3e}")

def viz_level(pyramid, level=0, cmap='viridis', size=(14,6), show_phase=True, clip_percentile=(1,99)):
    """
    Visualize one pyramid level: for each band show magnitude (and optionally phase).
    - pyramid: highpasses tuple/list (finest -> coarsest)
    - level: index (0 = finest)
    - clip_percentile: percentile range for contrast (tuple low, high)
    """
    arr = pyramid[level]
    h, w, bands = arr.shape
    mags = np.abs(arr)
    phases = np.angle(arr)
    # robust vmin/vmax per level
    vmin = np.percentile(mags, clip_percentile[0])
    vmax = np.percentile(mags, clip_percentile[1])
    # log scale for display
    mags_disp = np.log1p(mags)

    plt.figure(figsize=size)
    for b in range(bands):
        ax = plt.subplot(2 if show_phase else 1, bands, b+1)
        im = ax.imshow(mags_disp[:, :, b], cmap=cmap, vmin=np.log1p(vmin), vmax=np.log1p(vmax))
        ax.set_title(f"lvl{level} band{b} | mag")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    if show_phase:
        for b in range(bands):
            ax = plt.subplot(2, bands, bands + b + 1)
            im = ax.imshow(phases[:, :, b], cmap='twilight', vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"lvl{level} band{b} | phase")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    plt.suptitle(f"Pyramid level {level} (shape={arr.shape})", fontsize=14)
    plt.tight_layout()
    plt.show()

def viz_multiple_levels(pyramid, levels=None, cmap='viridis', size_per_level=(3, 3), clip_percentile=(1,99)):
    """
    Show magnitude images for selected levels in a grid:
    - rows = bands (6), columns = chosen levels (finest -> coarsest)
    """
    if levels is None:
        levels = list(range(len(pyramid)))
    # limit columns for readability
    max_cols = 6
    if len(levels) > max_cols:
        print(f"Showing first {max_cols} of {len(levels)} levels for readability.")
        levels = levels[:max_cols]
    bands = pyramid[0].shape[2]
    fig_h = bands * size_per_level[0]
    fig_w = len(levels) * size_per_level[1]
    fig, axes = plt.subplots(bands, len(levels), figsize=(fig_w, fig_h))
    for col_idx, lvl in enumerate(levels):
        arr = pyramid[lvl]
        mags = np.abs(arr)
        vmin = np.percentile(mags, clip_percentile[0])
        vmax = np.percentile(mags, clip_percentile[1])
        mags_disp = np.log1p(mags)
        for b in range(bands):
            ax = axes[b, col_idx] if bands > 1 else axes[col_idx]
            ax.imshow(mags_disp[:, :, b], cmap=cmap, vmin=np.log1p(vmin), vmax=np.log1p(vmax))
            if col_idx == 0:
                ax.set_ylabel(f"band {b}")
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(f"level {lvl}")
    plt.suptitle("Magnitude (log1p) across levels (bands rows, levels cols)")
    plt.tight_layout()
    plt.show()

def plot_band_across_levels(pyramid, band=0, cmap='magma', clip_percentile=(1,99)):
    """
    Plot the same band index across all levels (coarsest/finest horizontally).
    """
    levels = list(range(len(pyramid)))
    n = len(levels)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3))
    for i, lvl in enumerate(levels):
        arr = pyramid[lvl]
        mag = np.abs(arr[:, :, band])
        vmin = np.percentile(mag, clip_percentile[0])
        vmax = np.percentile(mag, clip_percentile[1])
        axs[i].imshow(np.log1p(mag), cmap=cmap, vmin=np.log1p(vmin), vmax=np.log1p(vmax))
        axs[i].set_title(f"lvl {lvl}")
        axs[i].axis('off')
    plt.suptitle(f"Band {band} across levels")
    plt.tight_layout()
    plt.show()


def plot_lowpass_image(lp, title="lowpass", cmap="viridis", log_scale=True, downsample=1, clip=(1,99)):
    """
    Simple image view for a lowpass array.
    - lp: numpy-compatible array (2D) or complex 2D
    - log_scale: show log1p(magnitude) to compress dynamic range
    - downsample: integer factor (1 = no downsample)
    - clip: percentile tuple (low, high) for contrast clipping
    """
    a = np.squeeze(np.asarray(lp))
    if a.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got shape {a.shape}")
    is_complex = np.iscomplexobj(a)
    mag = np.abs(a) if is_complex else a.astype(float)
    if downsample > 1:
        mag = mag[::downsample, ::downsample]
        if is_complex:
            phase = np.angle(a)[::downsample, ::downsample]
        else:
            phase = None
    else:
        phase = np.angle(a) if is_complex else None

    vmin = np.percentile(mag, clip[0])
    vmax = np.percentile(mag, clip[1])
    disp = np.log1p(mag) if log_scale else mag

    plt.figure(figsize=(6,6))
    plt.imshow(disp, cmap=cmap, vmin=np.log1p(vmin) if log_scale else vmin,
               vmax=np.log1p(vmax) if log_scale else vmax)
    plt.title(f"{title} (magnitude {'log1p' if log_scale else ''})")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.01)
    plt.show()

    if is_complex:
        plt.figure(figsize=(6,4))
        plt.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.title(f"{title} (phase)")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.01)
        plt.show()
# %%
# Parameters (edit these variables interactively before running the next cells)
# input_folder = os.path.join(os.path.dirname(__file__), 'sample_data')  # change path as needed
input_folder = "D:/AOFLIO_rates/AOFLIO-0035/AOFLIO004/phasor_images_g_s_intensity_LSC"

H = 1.0      # harmonic
tau = 0.25    # target tau,ns
flevel = 4   # number of filtering levels
ns = 12.5             # period of laser in ns (for 80MHz laser)
# ns = 12.820512820513  # period of laser in ns (for 78MHz laser)
save_outputs = False  # set False to skip writing files

# %%
# Prepare paths and check input files
start_time = time.time()
output_base_directory = os.path.dirname(os.path.abspath(__file__))
print(f"Output directory is set to: {output_base_directory}")

g_tif = os.path.join(input_folder, "G.tif")
s_tif = os.path.join(input_folder, "S.tif")
intensity_tif = os.path.join(input_folder, "intensity.tif")

if not all(map(os.path.exists, [g_tif, s_tif, intensity_tif])):
    raise FileNotFoundError("One or more of the required .tif files (G.tif, S.tif, intensity.tif) are missing in the input folder.")

# %%
# Compute Gc, Sc (phasor center) and process files
Gc, Sc = calculate_g_and_s(H, tau)
print(f"Gc={Gc:.4f}, Sc={Sc:.4f}")

G_combined = np.array([]).reshape(0, 0)
S_combined = np.array([]).reshape(0, 0)
I_combined = np.array([]).reshape(0, 0)
G_combined_unfil = np.array([]).reshape(0, 0)
S_combined_unfil = np.array([]).reshape(0, 0)
I_combined_unfil = np.array([]).reshape(0, 0)

file_paths = {"G": g_tif, "S": s_tif, "intensity": intensity_tif}
G_combined, S_combined, I_combined, coeffs = process_files(file_paths, G_combined, S_combined, I_combined, flevel)
G_combined_unfil, S_combined_unfil, I_combined_unfil = process_unfil_files(file_paths, G_combined_unfil, S_combined_unfil, I_combined_unfil)

# %% 
# Inspect attributes on a pyramid object and print shapes
def inspect_pyramid(pyr, name="pyr"):
    print(f"--- {name} ({type(pyr)}) ---")
    # list top-level attributes
    attrs = [a for a in dir(pyr) if not a.startswith("_")]
    print("attrs:", attrs)
    # try to show common fields
    for field in ("highpasses", "lowpass", "scale", "scales", "lowpass_image"):
        if hasattr(pyr, field):
            val = getattr(pyr, field)
            print(f" {field}: type={type(val)}", end="")
            try:
                if hasattr(val, "shape"):
                    print(", shape=", val.shape)
                elif isinstance(val, (list, tuple)):
                    print(f", len={len(val)}")
                    # for lists of arrays, show shapes of first few
                    for i, elt in enumerate(val[:4]):
                        print(f"   [{i}] type={type(elt)}, shape={getattr(elt,'shape',None)}")
                else:
                    print()
            except Exception:
                print(" (unable to introspect shape)")
    print()
realCoeffs = coeffs['real']
realHighPasses = realCoeffs.highpasses
realLowPass = realCoeffs.lowpass
realScales = realCoeffs.scales
imagCoeffs = coeffs['imag']
imagHighPasses = imagCoeffs.highpasses
imagLowPass = imagCoeffs.lowpass
imagScales = imagCoeffs.scales
intensityCoeffs = coeffs['intensity']
intensityHighPasses = intensityCoeffs.highpasses
intensityLowPass = intensityCoeffs.lowpass
intensityScales = intensityCoeffs.scales


# Example:
inspect_pyramid(realCoeffs, "realCoeffs")
inspect_pyramid(imagCoeffs, "imagCoeffs")
inspect_pyramid(intensityCoeffs, "intensityCoeffs")

# %%
# plot_lowpass_image(realLowPass, "real lowpass")
# plot_lowpass_image(imagLowPass, "imag lowpass")
# plot_lowpass_image(intensityLowPass, "intensity lowpass", log_scale=False)

# Let's iterate through the scales and call the plot_lowpass_image function for each
for i, scale in enumerate(realScales):
    plot_lowpass_image(scale, title=f"real scale {i}", log_scale=True)
for i, scale in enumerate(imagScales):
    plot_lowpass_image(scale, title=f"imag scale {i}", log_scale=True)
for i, scale in enumerate(intensityScales):
    plot_lowpass_image(scale, title=f"intensity scale {i}", log_scale=False)

# %%
# Utility: small table of level shapes & stats
print_pyramid_summary(realHighPasses, "realHighPasses")
print_pyramid_summary(imagHighPasses, "imagHighPasses")
print_pyramid_summary(intensityHighPasses, "intensityHighPasses")

# Example calls (uncomment to run)
viz_level(intensityHighPasses, level=0, show_phase=True)
# viz_multiple_levels(realHighPasses, levels=[0,1,2,3])
# viz_multiple_levels(imagHighPasses, levels=[0,1,2,3])
# viz_multiple_levels(intensityHighPasses, levels=[0,1,2,3])
# plot_band_across_levels(realHighPasses, band=2)
# %%
# Plot and save results

phasor_title = f'phasor_CWFlevels={flevel}.png'
npz_file_name = f'dataset_CWFlevels={flevel}.npz'
tiff_file_name = f'lifetime_CWFlevels={flevel}.tiff'
tiff_file_name_unfil = f'lifetime_unfiltered.tiff'

phasors_dir = os.path.join(output_base_directory, "phasors")
datasets_dir = os.path.join(output_base_directory, "datasets")
lifetime_images_dir = os.path.join(output_base_directory, "lifetime_images")
lifetime_images_unfil_dir = os.path.join(output_base_directory, "lifetime_images_unfiltered")

os.makedirs(phasors_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(lifetime_images_dir, exist_ok=True)
os.makedirs(lifetime_images_unfil_dir, exist_ok=True)

phasor_path = os.path.join(phasors_dir, phasor_title)
plot_combined_data(G_combined, S_combined, I_combined, phasor_path)

T = calculate_and_plot_lifetime(G_combined, S_combined, ns=ns,filter=True)
if save_outputs:
    tiff_path = os.path.join(lifetime_images_dir, tiff_file_name)
    tiff.imwrite(tiff_path, T)
    print(f"Lifetime image saved as {tiff_path}")

T_unfil = calculate_and_plot_lifetime(G_combined_unfil, S_combined_unfil,ns=ns, filter=False)
if save_outputs:
    tiff_path_unfil = os.path.join(lifetime_images_unfil_dir, tiff_file_name_unfil)
    tiff.imwrite(tiff_path_unfil, T_unfil)
    print(f"Unfiltered lifetime image saved as {tiff_path_unfil}")

npz_path = os.path.join(datasets_dir, npz_file_name)
if save_outputs:
    np.savez(npz_path, G=G_combined, S=S_combined, A=I_combined, T=T)
    print(f"NPZ dataset saved as {npz_path}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# %%
# Optional: display summary
print(f"Phasor image: {phasor_path}")
print(f"Dataset file: {npz_path}")
print(f"Lifetime image: {tiff_path if save_outputs else 'not saved'}")
