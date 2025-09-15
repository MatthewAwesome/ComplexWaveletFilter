#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line version of ComplexWaveletFilter.py
Prompts for input folder and parameters via command-line arguments.
"""


import os
import argparse
import time
import numpy as np
import tifffile as tiff
from ComplexWaveletFilter import (
    calculate_g_and_s,
    process_files,
    process_unfil_files,
    plot_combined_data,
    calculate_and_plot_lifetime
)

def parse_args():
    parser = argparse.ArgumentParser(description="Complex Wavelet Filter (command-line version)")
    parser.add_argument('--input-folder', type=str, required=True, help='Path to folder containing G.tif, S.tif, and intensity.tif')
    parser.add_argument('--harmonic', type=float, required=True, help='Harmonic (H) value')
    parser.add_argument('--tau', type=float, required=True, help='Target tau value (τ)')
    parser.add_argument('--flevel', type=int, required=True, help='Number of filtering levels (flevel)')
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    input_folder = args.input_folder
    H = args.harmonic
    tau = args.tau
    flevel = args.flevel

    output_base_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"Output directory is set to: {output_base_directory}")

    Gc, Sc = calculate_g_and_s(H, tau)

    g_tif = os.path.join(input_folder, "G.tif")
    s_tif = os.path.join(input_folder, "S.tif")
    intensity_tif = os.path.join(input_folder, "intensity.tif")

    if not all(map(os.path.exists, [g_tif, s_tif, intensity_tif])):
        raise FileNotFoundError("One or more of the required .tif files (G.tif, S.tif, intensity.tif) are missing in the input folder.")

    G_combined = np.array([]).reshape(0, 0)
    S_combined = np.array([]).reshape(0, 0)
    I_combined = np.array([]).reshape(0, 0)
    G_combined_unfil = np.array([]).reshape(0, 0)
    S_combined_unfil = np.array([]).reshape(0, 0)
    I_combined_unfil = np.array([]).reshape(0, 0)

    file_paths = {
        "G": g_tif,
        "S": s_tif,
        "intensity": intensity_tif
    }
    G_combined, S_combined, I_combined, _ = process_files(file_paths, G_combined, S_combined, I_combined, flevel)
    G_combined_unfil, S_combined_unfil, I_combined_unfil  = process_unfil_files(file_paths, G_combined_unfil, S_combined_unfil, I_combined_unfil)

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

    T = calculate_and_plot_lifetime(G_combined, S_combined)
    tiff_path = os.path.join(lifetime_images_dir, tiff_file_name)
    tiff.imwrite(tiff_path, T)

    T_unfil = calculate_and_plot_lifetime(G_combined_unfil, S_combined_unfil)
    tiff_path_unfil = os.path.join(lifetime_images_unfil_dir, tiff_file_name_unfil)
    tiff.imwrite(tiff_path_unfil, T_unfil)

    npz_path = os.path.join(datasets_dir, npz_file_name)
    np.savez(npz_path, G=G_combined, S=S_combined, A=I_combined, T=T)

    print(f"Phasor plot saved as {phasor_path}")
    print(f"NPZ dataset saved as {npz_path}")
    print(f"Lifetime image saved as {tiff_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script execution time: {elapsed_time:.2f} seconds")
