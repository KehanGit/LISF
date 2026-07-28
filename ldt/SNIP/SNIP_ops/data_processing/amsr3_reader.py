#!/usr/bin/env python3
# -----------------------BEGIN NOTICE -- DO NOT EDIT-----------------------
# NASA Goddard Space Flight Center
# Land Information System Framework (LISF)
# Version 7.8
#
# Copyright (c) 2026 United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All Rights Reserved.
# -------------------------END NOTICE -- DO NOT EDIT-----------------------
"""
SCRIPT: amsr3_reader.py

Script for reading AMSR3 data files using Xarray, Dask, and Pyresample.
Processes and saves files individually without merging.

REVISION HISTORY:
22 Jul 2026: Kehan Yang, Initial Specification.
"""

# Standard modules
from datetime import datetime, timedelta
import glob
import logging
import os
import time

# Third party modules
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from scipy.spatial import cKDTree

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SnowDepthPredictor')


class AMSR3DataProcessor:
    """Class for handling AMSR3 data"""

    def __init__(self, config=None):
        self.config = config
        self.target_resolution = self.config.target_resolution
        self.amsr3_files = []

    def process_l1r_data(self, target_datetime, max_retries=1):
        """Main processing pipeline for AMSR3 L1R data with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                logger.info("Processing attempt %s / %s for %s",
                            (attempt + 1), (max_retries + 1), target_datetime)

                # 1.1 Check available data
                available_files = self.check_available_data(target_datetime)
                output_filenames = self.generate_output_filename(
                    available_files)

                saved_files = []

                # 1.2 Process, resample, and save EACH file individually
                for file_path, out_file in zip(available_files,
                                               output_filenames):
                    logger.info("Starting processing for: %s",
                                os.path.basename(file_path))

                    # Read the raw data
                    tb_data = self.get_amsr3_l1r_dask(file_path)

                    # Resample the single file
                    resampled_data = self.resample_single_file(tb_data)

                    # Save to NetCDF
                    saved_path = self.save_to_nc_single(resampled_data,
                                                        out_file,
                                                        target_datetime)

                    if os.path.exists(saved_path):
                        saved_files.append(saved_path)
                    else:
                        raise Exception(
                            f"File not properly saved: {saved_path}")

                logger.info("Successfully processed and saved %s files.",
                            len(saved_files))
                return saved_files

            except Exception as e:
                logger.error("Attempt %s failed: %s", (attempt + 1), str(e))
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    raise
        return None

    def check_available_data(self, target_datetime):
        """Check for available AMSR3 files within 6-hour window"""
        start_time = target_datetime - \
                     timedelta(hours=self.config.time_window_hours)
        amsr3_path_root = self.config.project_path / \
                          self.config.amsr3_path
        logger.info('Search files in: %s', amsr3_path_root)

        hour = target_datetime.strftime('%H')

        # Determine which day to search based on the 6-hour window
        if hour == '00':
            search_datetime = target_datetime - timedelta(days=1)
        else:
            search_datetime = target_datetime

        year_str = search_datetime.strftime('%Y')
        month_str = search_datetime.strftime('%m')
        day_str = search_datetime.strftime('%d')

        amsr3_path = os.path.join(amsr3_path_root, year_str, month_str)

        all_files = []
        if os.path.exists(amsr3_path):
            search_pattern = os.path.join(
                amsr3_path,
                f"GGWAM3_{year_str}{month_str}{day_str}*.nc"
            )
            all_files.extend(glob.glob(search_pattern))
            logger.info("Found %s files in %s", len(all_files), amsr3_path)
        else:
            logger.warning("Directory does not exist: %s", amsr3_path)

        file_list = self._get_file_list(all_files, start_time, target_datetime)
        if not file_list:
            error_msg = (f"No AMSR3 descending files found for time window "
                         f"{start_time.strftime('%Y-%m-%d %H:%M')} to "
                         f"{target_datetime.strftime('%Y-%m-%d %H:%M')}")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Found %s AMSR3 descending files.", len(file_list))
        for file_path in file_list:
            logger.info("   - %s", file_path)

        self.amsr3_files = file_list
        return file_list

    def _get_file_list(self, all_files, start_time, target_datetime):
        """Internal function to filter AMSR3 files by time window and descending orbit."""
        file_list = []
        for file_path in all_files:
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                if len(parts) < 2:
                    continue
                time_and_orbit = parts[1]
                if len(time_and_orbit) < 13:
                    continue
                time_str = time_and_orbit[0:12]
                orbit = time_and_orbit[12]

                # Only keep Descending passes
                if orbit == 'D':
                    file_datetime = datetime.strptime(time_str, '%Y%m%d%H%M')
                    if start_time <= file_datetime <= target_datetime:
                        file_list.append(file_path)

            except (IndexError, ValueError) as e:
                logger.warning("Could not parse datetime from filename %s: %s",
                               filename, e)
                continue

        file_list.sort()
        return file_list

    def get_amsr3_l1r_dask(self, filename):
        """Read AMSR3 L1R data efficiently using Xarray and Dask."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cannot find file {filename}")

        try:
            ds = xr.open_dataset(filename, engine='h5netcdf', chunks='auto')

            tb_channels = {
                'tb_6v': "Tb_FOV06Ch06V_P89o",
                'tb_6h': "Tb_FOV06Ch06H_P89o",
                'tb_7v': "Tb_FOV06Ch07V_P89o",
                'tb_7h': "Tb_FOV06Ch07H_P89o",
                'tb_10vu': "Tb_FOV10Ch10uV_P89o", # extra 10.25 GHz channel
                'tb_10hu': "Tb_FOV10Ch10uH_P89o", # extra 10.26 GHz channel
                'tb_10v': "Tb_FOV10Ch10V_P89o", # same 10.65 GHz as AMSR2
                'tb_10h': "Tb_FOV10Ch10H_P89o", # same 10.65 GHz as AMSR2
                'tb_18v': "Tb_FOV10Ch18V_P89o",
                'tb_18h': "Tb_FOV10Ch18H_P89o",
                'tb_23v': "Tb_FOV23Ch23V_P89o",
                'tb_23h': "Tb_FOV23Ch23H_P89o",
                'tb_36v': "Tb_FOV36Ch36V_P89o",
                'tb_36h': "Tb_FOV36Ch36H_P89o",
                'tb_89v': "Tb_FOV36Ch89V_P89o",
                'tb_89h': "Tb_FOV36Ch89H_P89o",
                'tb_165v': "Tb_FOV36Ch165V_P89o", # no h pol for 165 GHz
                'tb_183v': "Tb_FOV36Ch183r7V_P89o",
                'tb_183h': "Tb_FOV36Ch183r7H_P89o"
            }

            results = {}

            # Dask array geometry extraction
            results['lat89'] = ds['Latitude_P89o'].data
            results['lon89'] = ds['Longitude_P89o'].data
            results['n89'], results['m89'] = ds['Latitude_P89o'].shape

            for key, amsr3_name in tb_channels.items():
                if amsr3_name in ds:
                    results[key] = ds[amsr3_name].data.astype(np.float32)
                else:
                    results[key] = da.zeros((results['n89'], results['m89']),
                                            dtype=np.float32)

            if 'ScanTimeTAI93' in ds:
                results['scan_time'] = ds['ScanTimeTAI93'].data

            return results

        except Exception as e:
            logger.error("Error reading AMSR3 file with xarray/dask: %s", e)
            raise

    def custom_idw_resample(self, src_lats, src_lons, src_data, tgt_lats,
                            tgt_lons, radius_m, neighbors=50):
        """Pure SciPy/NumPy Inverse Distance Weighting (IDW). Safe for Mac."""

        # 1. Filter out NaNs and invalid data
        valid = ~np.isnan(src_lats) & ~np.isnan(src_lons) & ~np.isnan(src_data)
        if not np.any(valid):
            return np.full(tgt_lats.shape, np.nan, dtype=np.float32)

        slat = np.deg2rad(src_lats[valid])
        slon = np.deg2rad(src_lons[valid])
        sdata = src_data[valid]

        # 2. Convert Source to 3D Cartesian (meters) for accurate KDTree distance
        R = 6371000.0  # Radius of Earth in meters
        sx = R * np.cos(slat) * np.cos(slon)
        sy = R * np.cos(slat) * np.sin(slon)
        sz = R * np.sin(slat)

        # 3. Convert Target to 3D Cartesian
        tlat = np.deg2rad(tgt_lats.ravel())
        tlon = np.deg2rad(tgt_lons.ravel())
        tx = R * np.cos(tlat) * np.cos(tlon)
        ty = R * np.cos(tlat) * np.sin(tlon)
        tz = R * np.sin(tlat)

        # 4. Build KDTree and find nearest neighbors within the specific radius
        tree = cKDTree(np.column_stack([sx, sy, sz]))

        # Returns distances and index positions of the nearest points
        dists, idxs = tree.query(
            np.column_stack([tx, ty, tz]),
            k=neighbors,
            distance_upper_bound=radius_m
        )

        # 5. Inverse Distance Weighting Math
        valid_mask = idxs < len(
            sdata)  # cKDTree sets out-of-bounds neighbors to len(data)
        sdata_flat = sdata.flatten()
        # Gather the temperatures (use 0 for invalid to keep array shapes happy, we mask it out next)
        safe_idxs = np.where(valid_mask, idxs, 0)
        vals = sdata_flat[safe_idxs]

        # Weights = 1 / distance (add 1e-6 to prevent dividing by zero)
        weights = np.where(valid_mask, 1.0 / (dists + 1e-6), 0.0)

        sum_weights = np.sum(weights, axis=1)
        # print(f"Weights shape: {weights.shape}, Vals shape: {vals.shape}")
        weighted_vals = np.sum(weights * vals, axis=1)

        # Suppress the harmless divide-by-zero warning
        with np.errstate(divide='ignore', invalid='ignore'):
            out_flat = np.where(sum_weights > 0, weighted_vals / sum_weights,
                                np.nan)

        return out_flat.reshape(tgt_lats.shape).astype(np.float32)

    def resample_single_file(self, data):
        """
        Resamples a single AMSR3 swath dataset to the fixed ARFS/AF Grid using SciPy IDW.
        """
        logger.info(
            "Resampling single dataset using Custom SciPy IDW to fixed AF Grid...")

        # 1. Define the exact AF Grid dimensions
        width = 2560
        height = 1920

        # Generate the 1D arrays for the NetCDF coordinates
        # Matching Pyresample's extent: (-180.0, -90.0, 180.0, 90.0)
        lons_1d = np.linspace(-180.0, 180.0, width)
        lats_1d = np.linspace(90.0, -90.0,
                              height)  # Starts at 90, goes down to -90

        # Generate the 2D grid needed for the KDTree math
        target_lon_2d, target_lat_2d = np.meshgrid(lons_1d, lats_1d)

        resampled_result = {
            'lat': lats_1d,
            'lon': lons_1d,
            # <--- FIXED: Make sure this is lons_1d, not lats_1d!
        }

        # Define scientifically accurate search radii (in meters)
        radius_map = {
            'tb_6': 50000,
            'tb_7': 50000,
            'tb_10': 30000,
            'tb_18': 20000,
            'tb_23': 20000,
            'tb_36': 15000,
            'tb_89': 15000,
            'tb_165': 15000,
            'tb_183': 15000
        }

        tb_channels = [k for k in data.keys() if k.startswith('tb_')]

        # Compute raw coordinates from the file
        swath_lats = data['lat89'].compute()
        swath_lons = data['lon89'].compute()

        for var in tb_channels:
            raw_data = data[var].compute()

            # Dynamically select the correct search radius for this channel
            base_channel = var.split('v')[0].split('h')[0]
            search_radius = radius_map.get(base_channel, 20000)

            # Resample using our custom IDW function
            resampled = self.custom_idw_resample(
                src_lats=swath_lats,
                src_lons=swath_lons,
                src_data=raw_data,
                tgt_lats=target_lat_2d,
                tgt_lons=target_lon_2d,
                radius_m=search_radius,
                neighbors=8
                # Grabs up to 8 overlapping pixels within the radius to average
            )

            resampled_result[var] = resampled

        return resampled_result

    def save_to_nc_single(self, resampled_data, output_filename,
                          target_datetime):
        """
        Save a single resampled AMSR3 dataset to a NetCDF file.
        """
        try:
            logger.info("Saving resampled file to %s", output_filename)

            lat_1d = resampled_data['lat']
            lon_1d = resampled_data['lon']
            time_coord = pd.to_datetime([target_datetime])

            data_vars = {}
            tb_keys = [k for k in resampled_data.keys() if k.startswith('tb_')]

            for var_name in tb_keys:
                data_vars[var_name] = (
                    ['time', 'y', 'x'],
                    resampled_data[var_name][np.newaxis, :, :].astype(
                        np.float32)
                )

            ds_xr = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'time': time_coord,
                    'y': lat_1d,
                    'x': lon_1d
                }
            )

            # Save to nc file
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            ds_xr.to_netcdf(output_filename)

            logger.info("Successfully saved NetCDF file: %s", output_filename)
            return output_filename

        except Exception as e:
            logger.error("Error saving NetCDF file %s: %s", output_filename, e)
            raise

    def generate_output_filename(self, available_files, output_dir=None):
        """Generates a list of NetCDF output filenames based on input files."""
        if output_dir is None:
            output_dir = getattr(self.config, 'amsr3_resample_path')

        full_output_dir = self.config.project_path / output_dir
        os.makedirs(full_output_dir, exist_ok=True)

        outfiles = []
        for f in available_files:
            basename = os.path.basename(f)
            outfiles.append(full_output_dir / basename)

        return outfiles