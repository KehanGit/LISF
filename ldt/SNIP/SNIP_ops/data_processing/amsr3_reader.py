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
from pyresample import geometry, kd_tree

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

    def process_l1r_data(self, target_datetime, max_retries=3):
        """Main processing pipeline for AMSR3 L1R data with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                logger.info("Processing attempt %s / %s for %s",
                            (attempt + 1), (max_retries + 1), target_datetime)

                # 1.1 Check available data
                available_files = self.check_available_data(target_datetime)

                # 1.2 Process each file using Dask-backed Xarray
                processed_data = []
                for file_path in available_files:
                    tb_data = self.get_amsr3_l1r_dask(file_path)
                    processed_data.append(tb_data)

                # 1.3 Combine and resample datasets using Pyresample
                combined_data = self.combine_data(processed_data)

                # 1.4 Save to Zarr
                output_filename = self.generate_output_filename(target_datetime)
                saved_file = self.save_to_zarr(combined_data, output_filename,
                                               target_datetime)

                if os.path.exists(saved_file):
                    logger.info("Successfully processed and saved to Zarr: %s",
                                saved_file)
                    return combined_data, saved_file

                raise Exception(f"File not properly saved: {saved_file}")

            except Exception as e:
                logger.error("Attempt %s failed: %s", (attempt + 1), str(e))
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    raise
        return None, None

    def check_available_data(self, target_datetime):
        """Check for available AMSR3 files within 6-hour window"""
        start_time = target_datetime - \
                     timedelta(hours=self.config.time_window_hours)
        amsr2_path_root = self.config.project_path / \
                          self.config.amsr2_path
        logger.info('Search files in: %s', amsr2_path_root)

        year_str = target_datetime.strftime('%Y')
        day_of_year = target_datetime.strftime('%j')
        month = target_datetime.strftime('%m')
        hour = target_datetime.strftime('%H')

        amsr2_path = ''
        if hour == '00':
            prev_datetime = target_datetime - timedelta(days=1)
            year_str = prev_datetime.strftime('%Y')
            month = prev_datetime.strftime('%m')
            day_of_year = prev_datetime.strftime('%j')

        if self.config.AMSR2_source == 'NOAA':
            amsr2_path = os.path.join(amsr2_path_root, year_str, day_of_year)
        elif self.config.AMSR2_source == 'JAXA':
            amsr2_path = os.path.join(amsr2_path_root, year_str, month)
        else:
            logger.error("Wrong Source provided (either NOAA or JAXA)")

        all_files = []
        if os.path.exists(amsr2_path):
            all_files.extend(glob.glob(os.path.join(amsr2_path, "*.h5")))
            logger.info("Found %s files", len(all_files))
        else:
            logger.warning("Path does not exist: %s", amsr2_path)

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
        """Internal function to get file list."""
        file_list = []
        for file_path in all_files:
            try:
                filename = os.path.basename(file_path)

                if self.config.AMSR2_source == 'JAXA':
                    if 'L1SGRTBR' not in filename:
                        continue
                    orbit = filename.split('_')[2][3]
                    if orbit == 'D':
                        time_str = filename.split('_')[1]
                        if len(time_str) == 12:
                            file_datetime = datetime.strptime(time_str,
                                                              '%Y%m%d%H%M')
                        else:
                            continue
                        if start_time <= file_datetime <= target_datetime:
                            file_list.append(file_path)

                elif self.config.AMSR2_source == 'NOAA':
                    if 'L1DLRTBR' not in filename:
                        continue
                    time_str = filename.split('_')[1][0:12]
                    if len(time_str) == 12:
                        file_datetime = datetime.strptime(time_str,
                                                          '%Y%m%d%H%M')
                    else:
                        continue
                    if start_time <= file_datetime <= target_datetime:
                        file_list.append(file_path)

            except (IndexError, ValueError) as e:
                continue

        file_list.sort()
        return file_list

    def get_amsr3_l1r_dask(self, filename):
        """Read AMSR3 L1R data efficiently using Xarray and Dask."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cannot find file {filename}")

        try:
            ds = xr.open_dataset(filename, engine='h5netcdf',
                                 chunks={'time': 1000})

            tb_channels = {
                'tb_6v': "Tb_FOV06Ch06V_P89o",
                'tb_6h': "Tb_FOV06Ch06H_P89o",
                'tb_7v': "Tb_FOV06Ch07V_P89o",
                'tb_7h': "Tb_FOV06Ch07H_P89o",
                'tb_10v': "Tb_FOV10Ch10V_P89o",
                'tb_10h': "Tb_FOV10Ch10H_P89o",
                'tb_18v': "Tb_FOV10Ch18V_P89o",
                'tb_18h': "Tb_FOV10Ch18H_P89o",
                'tb_23v': "Tb_FOV23Ch23V_P89o",
                'tb_23h': "Tb_FOV23Ch23H_P89o",
                'tb_36v': "Tb_FOV36Ch36V_P89o",
                'tb_36h': "Tb_FOV36Ch36H_P89o",
                'tb_89v': "Tb_FOV36Ch89V_P89o",
                'tb_89h': "Tb_FOV36Ch89H_P89o",
                'tb_165v': "Tb_FOV36Ch165V_P89o",
                # no 165h channel in the nc file
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
                results['tb_time'] = ds['ScanTimeTAI93'].data

            return results

        except Exception as e:
            logger.error("Error reading AMSR3 file with xarray/dask: %s", e)
            raise

    def combine_data(self, processed_data):
        """
        Dask-parallelized combining and resampling using pyresample.
        Replaces slow Scipy KDTree and BivariateSpline methods.
        """
        if not processed_data:
            logger.warning("No processed data to combine")
            return {}

        logger.info("Combining %s AMSR3 datasets using Pyresample",
                    len(processed_data))

        # Define grid bounds natively without pulling massive arrays into memory
        lat_min, lat_max = 90.0, -90.0
        lon_min, lon_max = 180.0, -180.0

        for data in processed_data:
            # We compute just the min/max of the Dask arrays to find the bounding box
            batch_lat_min = da.min(data['lat89']).compute()
            batch_lat_max = da.max(data['lat89']).compute()
            batch_lon_min = da.min(data['lon89']).compute()
            batch_lon_max = da.max(data['lon89']).compute()

            lat_min = min(lat_min, batch_lat_min)
            lat_max = max(lat_max, batch_lat_max)
            lon_min = min(lon_min, batch_lon_min)
            lon_max = max(lon_max, batch_lon_max)

        # Build Pyresample Target Area Definition (Plate Carree / Equirectangular for lat/lon)
        area_id = 'target_grid'
        proj_id = 'latlong'
        proj_dict = {'proj': 'longlat', 'datum': 'WGS84'}

        # Calculate width and height based on config resolution
        width = int(np.ceil((lon_max - lon_min) / self.target_resolution))
        height = int(np.ceil((lat_max - lat_min) / self.target_resolution))
        area_extent = (lon_min, lat_min, lon_max, lat_max)

        target_def = geometry.AreaDefinition(area_id, 'Target Grid', proj_id,
                                             proj_dict, width, height,
                                             area_extent)

        # Store output coordinates for the final Zarr file
        lons_1d, lats_1d = target_def.get_proj_vectors()

        combined_result = {
            'lat': lats_1d,
            'lon': lons_1d,
            'n': height,
            'm': width
        }

        # List of variables to resample
        tb_channels = [k for k in processed_data[0].keys() if
                       k.startswith('tb_')]

        # Initialize target arrays for the final merged result
        for var in tb_channels:
            combined_result[var] = np.full((height, width), np.nan,
                                           dtype=np.float32)

        # Resample each orbit using Pyresample KDTree
        for idx, data in enumerate(processed_data):
            logger.info("Resampling dataset %s/%s", idx + 1,
                        len(processed_data))

            # Pyresample requires numpy arrays, so we compute the chunk here
            swath_lats = data['lat89'].compute()
            swath_lons = data['lon89'].compute()
            swath_def = geometry.SwathDefinition(lons=swath_lons,
                                                 lats=swath_lats)

            for var in tb_channels:
                if var in data:
                    raw_data = data[var].compute()

                    # Dask-friendly KDTree Resampling
                    # Radius of influence (meters): AMSR 6GHz FOV is ~50km
                    resampled = kd_tree.resample_nearest(
                        source_geo_def=swath_def,
                        data=raw_data,
                        target_geo_def=target_def,
                        radius_of_influence=50000,
                        fill_value=np.nan
                    )

                    # Merge logic: if combined is NaN, overwrite it with new data
                    # If both have data, take the mean
                    existing_data = combined_result[var]

                    mask_both_valid = ~np.isnan(existing_data) & ~np.isnan(
                        resampled)
                    mask_new_only = np.isnan(existing_data) & ~np.isnan(
                        resampled)

                    existing_data[mask_new_only] = resampled[mask_new_only]
                    existing_data[mask_both_valid] = (existing_data[
                                                          mask_both_valid] +
                                                      resampled[
                                                          mask_both_valid]) / 2.0

                    combined_result[var] = existing_data

        logger.info("Successfully combined and regridded AMSR3 datasets.")
        return combined_result

    def save_to_zarr(self, combined_data, output_path, target_datetime):
        """
        Save combined AMSR3 dataset to a Zarr store using Xarray.
        """
        try:
            logger.info("Saving combined dataset to %s", output_path)

            lat_1d = combined_data['lat']
            lon_1d = combined_data['lon']
            time_coord = pd.to_datetime([target_datetime])

            data_vars = {}
            tb_keys = [k for k in combined_data.keys() if k.startswith('tb_')]

            for var_name in tb_keys:
                data_vars[var_name] = (
                    ['time', 'y', 'x'],
                    combined_data[var_name][np.newaxis, :, :].astype(np.float32)
                )

            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'time': time_coord,
                    'y': lat_1d,
                    'x': lon_1d
                }
            )

            # Apply Dask chunking before saving
            ds = ds.chunk({'time': 1, 'y': 256, 'x': 256})

            # Save to Zarr
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ds.to_zarr(output_path, mode='w', consolidated=True)

            logger.info("Successfully saved Zarr store: %s", output_path)
            return output_path

        except Exception as e:
            logger.error("Error saving Zarr store: %s", e)
            raise

    def generate_output_filename(self, target_datetime, output_dir=None):
        if output_dir is None:
            output_dir = getattr(self.config, 'amsr2_merge_path',
                                 './amsr3_merge_path')

        full_output_dir = self.config.project_path / output_dir
        timestamp_str = target_datetime.strftime('%Y%m%d%H%M')

        filename = f'AMSR3_L1R_combined_{timestamp_str}.zarr'
        return full_output_dir / filename