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
                            (attempt + 1), (max_retries + 1),
                            target_datetime)

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

                    # get descending orbit only
                    tb_data_D = self.get_descending_orbit(tb_data)

                    # Resample the single file
                    resampled_data = self.resample_single_file(tb_data_D)

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
                logger.error("Attempt %s failed: %s",
                             (attempt + 1), str(e))
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
        day_of_year_str = search_datetime.strftime("%j")

        amsr3_path = os.path.join(amsr3_path_root, year_str, day_of_year_str)

        all_files = []
        if os.path.exists(amsr3_path):
            search_pattern = os.path.join(
                amsr3_path,
                f"GGWAM3_{year_str}{month_str}{day_str}*.nc"
            )
            all_files.extend(glob.glob(search_pattern))
            logger.info("Found %s files in %s",
                        len(all_files), amsr3_path)
        else:
            logger.warning("Directory does not exist: %s",
                           amsr3_path)

        file_list = self._get_file_list(all_files, start_time, target_datetime)
        if not file_list:
            error_msg = (f"No AMSR3 descending files found for time window "
                         f"{start_time.strftime('%Y-%m-%d %H:%M')} to "
                         f"{target_datetime.strftime('%Y-%m-%d %H:%M')}")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Found %s AMSR3 descending files.",
                    len(file_list))
        for file_path in file_list:
            logger.info("   - %s", file_path)

        self.amsr3_files = file_list
        return file_list

    def get_descending_orbit(self, ds):
        """Extract all segments where latitude is decreasing"""
        # Get latitude values
        logger.info('Process to get descending orbit for data from NOAA')
        if 'lat89' in ds.keys():
            lat_values = ds['lat89']
        else:
            lat_values = ds['latitude']
        # Handle 2D latitude arrays
        if hasattr(lat_values, 'ndim') and lat_values.ndim == 2:
            # Use the first row to find the pattern (all rows should be
            # the same)
            lat_row = lat_values[:, 0]  # first row
            lat_diff = np.diff(lat_row)
            is_decreasing = lat_diff < 0
            decreasing_mask = np.zeros(len(lat_row), dtype=bool)
            decreasing_indices = np.where(is_decreasing)[0]
            # Mark both start and end points
            decreasing_mask[decreasing_indices] = True  # Start points
            decreasing_mask[decreasing_indices + 1] = True  # End points
        else:
            # Original 1D case
            lat_diff = np.diff(lat_values)
            is_decreasing = lat_diff < 0
            decreasing_mask = np.zeros(len(lat_values), dtype=bool)
            decreasing_indices = np.where(is_decreasing)[0]
            decreasing_mask[decreasing_indices] = True
            decreasing_mask[decreasing_indices + 1] = True

        # Get final indices
        final_indices = np.where(decreasing_mask)[0]
        ## Apply to all arrays along the latitude axis
        ds_result = {}
        for var_name, var_array in ds.items():
            if hasattr(var_array, 'ndim') and var_array.ndim == 2 \
               and var_array.shape == lat_values.shape:
                ds_result[var_name] = var_array[final_indices, :]
            else:
                ds_result[var_name] = var_array
        return ds_result


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

                # Get files within the 6 hrs time window

                file_datetime = datetime.strptime(time_str,
                                                  '%Y%m%d%H%M')
                if start_time <= file_datetime <= target_datetime:
                    file_list.append(file_path)

            except (IndexError, ValueError) as e:
                logger.warning(
                    "Could not parse datetime from filename %s: %s",
                    filename, e)
                continue

        file_list.sort()
        return file_list

    def apply_flags(self, tb_data, land_percent, n, m):
        """
        Generate snow and precipitation flags based on TB values.

        Parameters:
        -----------
        tb_data : dict
            Dictionary containing TB arrays
        land_percent : array_like
            Land/water fraction array
        n, m : int
            Array dimensions

        Returns:
        --------
        tuple : (rain, cold_deserts, frozen_ground, glacier) flag arrays
        """

        rain = np.zeros((n, m), dtype=np.int32)
        cold_deserts = np.zeros((n, m), dtype=np.int32)
        frozen_ground = np.zeros((n, m), dtype=np.int32)
        glacier = np.zeros((n, m), dtype=np.int32)
        snow = np.zeros((n, m), dtype=np.int32)
        precip = np.zeros((n, m), dtype=np.int32)

        # Get required TB channels
        tb_18v = tb_data.get('tb_18v', np.zeros((n, m)))
        tb_18h = tb_data.get('tb_18h', np.zeros((n, m)))
        tb_23v = tb_data.get('tb_23v', np.zeros((n, m)))
        tb_36v = tb_data.get('tb_36v', np.zeros((n, m)))
        tb_89v = tb_data.get('tb_89v', np.zeros((n, m)))

        logger.info("Generating rain, cold deserts, frozen ground,"
                       " and glacier flags using NOAA method;"
                       "Generating snow and precip flags using Rajat method ...")

        for i in range(n):
            for j in range(m):
                # Only process data over land (land_percent >= 50)
                if land_percent[i, j] >= 50:
                    # Check if all required TB values are valid (> 0)
                    if (tb_18v[i, j] > 0 and tb_18h[i, j] > 0 and
                            tb_23v[i, j] > 0 and tb_89v[i, j] > 0 and
                            tb_36v[i, j] > 0):

                        # Apply NOAA's method in flag rain, cold deserts,
                        # frozen ground and glacier (Grody's SCA algorithm)
                        scat = tb_23v[i, j] - tb_89v[i, j]
                        sc37 = tb_18v[i, j] - tb_36v[i, j]
                        pd19 = tb_18v[i, j] - tb_18h[i, j]  # tt18
                        scx = tb_36v[i, j] - tb_89v[i, j]
                        scat = max(scat, sc37)
                        tt = (165 + 0.49 * tb_89v[i, j])
                        if ((tb_23v[i, j] >= 254.0 and scat <= 2.0) or
                                tb_23v[i, j] >= 258.0 or tb_23v[i, j] >= tt):
                            rain[i, j] = 1
                        else:
                            rain[i, j] = 0
                        if pd19 >= 18.0 and sc37 <= 10.0 and scx <= 10.0:
                            cold_deserts[i, j] = 1
                        else:
                            cold_deserts[i, j] = 0
                        if scat <= 6.0 and pd19 >= 8.0:
                            frozen_ground[i, j] = 1
                        else:
                            frozen_ground[i, j] = 0
                        if tb_23v[i, j] <= 210.0 or \
                           (tb_23v[i, j] <= 229.0 and pd19 >= 23.0):
                            glacier[i, j] = 1
                        else:
                            glacier[i, j] = 0
                        # Ehsan's method in classify snow and precipitation
                        # Calculate scattering index and polarization difference
                        sil = (451.88 - 0.44 * tb_18v[i, j] - 1.775 * tb_23v[i, j] +
                               0.00574 * tb_23v[i, j] ** 2 - tb_89v[i, j])
                        tt18 = tb_18v[i, j] - tb_18h[i, j]
                        if sil > 10:
                            if (tb_23v[i, j] <= 264.0 and
                                    tb_23v[i, j] <= (175.0 + 0.49 * tb_89v[i, j])):
                                # Snow branch
                                snow[i, j] = 1
                                # Additional snow checks
                                if (tt18 >= 18 and
                                        (tb_18v[i, j] - tb_36v[i, j]) <= 10 and
                                        (tb_36v[i, j] - tb_89v[i, j]) <= 10):
                                    snow[i, j] = 0
                                if (tt18 >= 8 and
                                        (tb_18v[i, j] - tb_36v[i, j]) <= 2 and
                                        (tb_23v[i, j] - tb_89v[i, j]) <= 6):
                                    snow[i, j] = 1
                            else:
                                # Precipitation branch
                                snow[i, j] = 0
                                precip[i, j] = 1
                                # Additional precip checks
                                if tt18 > 20:
                                    precip[i, j] = 0
                                if tb_89v[i, j] > 253 and tt18 > 7:
                                    precip[i, j] = 0

        logger.info("Rain, cold_deserts, frozen_ground, "
                       "glacier, snow, and precip flags generated")
        return rain, cold_deserts, frozen_ground, glacier, snow, precip

    def get_amsr3_l1r_dask(self, filename):
        """Read AMSR3 L1R data efficiently using Xarray and Dask."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Cannot find file {filename}")

        try:
            ds = xr.open_dataset(filename,
                                 engine='h5netcdf',
                                 chunks='auto')

            # TODO v7.9 AMSR3 ML model
            tb_channels = {
                'tb_6v': "Tb_FOV06Ch06V_P89o",
                'tb_6h': "Tb_FOV06Ch06H_P89o",
                'tb_7v': "Tb_FOV06Ch07V_P89o",
                'tb_7h': "Tb_FOV06Ch07H_P89o",
                # 'tb_10vu': "Tb_FOV10Ch10uV_P89o",
                # 'tb_10hu': "Tb_FOV10Ch10uH_P89o",
                'tb_10v': "Tb_FOV10Ch10V_P89o",
                'tb_10h': "Tb_FOV10Ch10H_P89o",
                'tb_18v': "Tb_FOV10Ch18V_P89o",
                'tb_18h': "Tb_FOV10Ch18H_P89o",
                'tb_23v': "Tb_FOV23Ch23V_P89o",
                'tb_23h': "Tb_FOV23Ch23H_P89o",
                'tb_36v': "Tb_FOV36Ch36V_P89o",
                'tb_36h': "Tb_FOV36Ch36H_P89o",
                'tb_89v': "Tb_FOV36Ch89V_P89o",
                'tb_89h': "Tb_FOV36Ch89H_P89o"
                # 'tb_165v': "Tb_FOV36Ch165V_P89o",
                # 'tb_183v': "Tb_FOV36Ch183r7V_P89o",
                # 'tb_183h': "Tb_FOV36Ch183r7H_P89o"
            }

            results = {}

            # Dask array geometry extraction
            results['lat89'] = ds['Latitude_P89o'].data
            results['lon89'] = ds['Longitude_P89o'].data
            results['n89'], results['m89'] = ds['Latitude_P89o'].shape

            for key, amsr3_name in tb_channels.items():
                if amsr3_name in ds:
                    results[key] = ds[amsr3_name].data.astype(np.float32)

                    # Look for the matching _Quality array
                    qual_name = amsr3_name + '_Quality'
                    if qual_name in ds:
                        results[key + '_quality'] = ds[qual_name].data.astype(
                            np.float32)

            # Grab the highest resolution land percentage
            if 'LandAreaPercent_FOV36_P89o' in ds:
                results['land_percent'] = ds[
                    'LandAreaPercent_FOV36_P89o'].data.astype(np.float32)

            if 'ScanDataQuality' in ds:
                scan_qual_1d = ds['ScanDataQuality'].data.astype(np.float32)
                scan_qual_2d = scan_qual_1d[:, np.newaxis]
                results['scan_quality'] = np.broadcast_to(scan_qual_2d, (
                results['n89'], results['m89']))


            if 'ScanTimeTAI93' in ds:
                results['scan_time'] = ds['ScanTimeTAI93'].data

            return results

        except Exception as e:
            logger.error("Error reading AMSR3 file with xarray/dask: %s",
                         e)
            raise

    def get_spatial_mapping(self, src_lats, src_lons, tgt_lats, tgt_lons):
        """Builds the KDTree and finds neighbors ONCE for the entire file."""
        logger.info(
            "Building KDTree and mapping coordinates ...")

        # 1. Filter valid coordinates (shared across all channels)
        valid_coords = ~np.isnan(src_lats) & ~np.isnan(src_lons) & (
                    src_lats >= -90) & (src_lats <= 90)

        slat = src_lats[valid_coords]
        slon = src_lons[valid_coords]

        # 2. Convert to 3D Cartesian
        R = 6371228.0
        sx = R * np.cos(np.deg2rad(slat)) * np.cos(np.deg2rad(slon))
        sy = R * np.cos(np.deg2rad(slat)) * np.sin(np.deg2rad(slon))
        sz = R * np.sin(np.deg2rad(slat))

        tx = R * np.cos(np.deg2rad(tgt_lats.ravel())) * np.cos(
            np.deg2rad(tgt_lons.ravel()))
        ty = R * np.cos(np.deg2rad(tgt_lats.ravel())) * np.sin(
            np.deg2rad(tgt_lons.ravel()))
        tz = R * np.sin(np.deg2rad(tgt_lats.ravel()))

        # 3. Build and Query KDTree
        tree = cKDTree(np.column_stack([sx, sy, sz]))
        dists, idxs = tree.query(np.column_stack([tx, ty, tz]), k=50,
                                 distance_upper_bound=20000.0)

        return dists, idxs, valid_coords

    def apply_idw(self, dists, idxs, valid_coords, src_data, tgt_shape,
                  src_quality=None):
        """Applies the IDW math instantly using the
        pre-calculated KDTree mapping."""
        # Align data with the valid coordinates used to build the tree
        sdata = src_data[valid_coords]
        if src_quality is not None:
            squal = src_quality[valid_coords]

        # Find which target pixels actually got valid neighbors
        valid_tree_mask = idxs < len(sdata)
        safe_idxs = np.where(valid_tree_mask, idxs, 0)

        # Extract values
        vals = sdata[safe_idxs]

        # Filter: Must be a valid neighbor AND physically valid TB
        valid_tb_mask = valid_tree_mask & (vals > 50.0) & (vals < 400.0)

        # Quality filtering (0 = Good)
        if src_quality is not None:
            q_vals = squal[safe_idxs]
            valid_tb_mask &= (q_vals == 0)

        # Apply IDW Math
        weights = np.where(valid_tb_mask, 1.0 / (dists + 1e-6), 0.0)
        sum_weights = np.sum(weights, axis=1)
        weighted_vals = np.sum(weights * np.where(valid_tb_mask, vals, 0),
                               axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            out_flat = np.where(sum_weights > 0, weighted_vals / sum_weights,
                                np.nan)

        return out_flat.reshape(tgt_shape).astype(np.float32)

    def resample_single_file(self, data):
        """
        Resamples using fixed 20km radius - optimized!
        Matches Fortran exactly: Resamples TBs and Land separately.
        Ocean pixels are PRESERVED here so the ML model can filter them later.
        """
        logger.info("Resampling using IDW Method (20km radius)...")

        width, height = 2560, 1920
        tgt_shape = (height, width)
        lons_1d = np.linspace(-180.0, 180.0, width)
        lats_1d = np.linspace(90.0, -90.0, height)
        target_lon_2d, target_lat_2d = np.meshgrid(lons_1d, lats_1d)

        resampled_result = {'lat': lats_1d, 'lon': lons_1d}
        tb_channels = [k for k in data.keys() if
                       k.startswith('tb_') and not k.endswith('_quality')]

        # 1. Compute Coordinates ONCE (Forcing float32 to prevent Xarray list errors)
        swath_lats = np.asarray(data['lat89'], dtype=np.float32).flatten()
        swath_lons = np.asarray(data['lon89'], dtype=np.float32).flatten()

        # 2. Build the Spatial Mapping ONCE
        dists, idxs, valid_coords = self.get_spatial_mapping(
            swath_lats, swath_lons, target_lat_2d, target_lon_2d
        )

        # Grab Scan Quality ONCE
        if 'scan_quality' in data:
            global_scan_quality = np.nan_to_num(
                np.asarray(data['scan_quality'], dtype=np.float32).flatten(),
                nan=0.0)
        else:
            global_scan_quality = None

        # 3. Apply the mapping to all TB channels INSTANTLY
        for var in tb_channels:
            raw_data = np.asarray(data[var], dtype=np.float32).flatten()

            # Fetch the matching channel quality array
            qual_key = var + '_quality'
            if qual_key in data:
                raw_quality = np.nan_to_num(
                    np.asarray(data[qual_key], dtype=np.float32).flatten(),
                    nan=0.0)
            else:
                raw_quality = np.zeros_like(raw_data)

            # --- COMBINE SENSOR QUALITY ONLY ---
            if global_scan_quality is not None:
                combined_quality = np.maximum(raw_quality, global_scan_quality)
            else:
                combined_quality = raw_quality

            resampled = self.apply_idw(
                dists, idxs, valid_coords, raw_data, tgt_shape,
                src_quality=combined_quality
            )
            resampled_result[var] = resampled

        # 4. Resample the Land Percentage using IDW
        if 'land_percent' in data:
            raw_land = np.asarray(data['land_percent'],
                                  dtype=np.float32).flatten()
            resampled_land = self.apply_idw(
                dists, idxs, valid_coords, raw_land, tgt_shape, src_quality=None
            )
            resampled_result['land_percent'] = resampled_land

        # --- Swath Mask (Ensure Identical Edges) ---
        master_mask = np.ones((height, width), dtype=bool)

        # A pixel must be valid in ALL channels to survive
        # (This cuts off the empty space outside the swath, but ignores whether it is land or ocean)
        for var in tb_channels:
            master_mask &= ~np.isnan(resampled_result[var])

        # 5. Apply the identical edge mask to ALL channels and land_percent
        for var in tb_channels:
            resampled_result[var] = np.where(master_mask, resampled_result[var],
                                             np.nan)

        if 'land_percent' in resampled_result:
            resampled_result['land_percent'] = np.where(master_mask,
                                                        resampled_result[
                                                            'land_percent'],
                                                        np.nan)

        return resampled_result

    def save_to_nc_single(self, resampled_data, output_filename,
                          target_datetime):
        """
        Save a single resampled AMSR3 dataset to a NetCDF file with GIS projection.
        """
        import rioxarray  # Ensure this is imported for the CRS tagging

        try:
            logger.info("Saving resampled file to %s", output_filename)

            lat_1d = resampled_data['lat']
            lon_1d = resampled_data['lon']
            time_coord = pd.to_datetime([target_datetime])

            data_vars = {}

            # Save TB channels
            keys_to_save = [k for k in resampled_data.keys() if
                            k not in ['lat', 'lon']]

            for var_name in keys_to_save:
                data_vars[var_name] = (
                    ['time', 'lat', 'lon'],
                    resampled_data[var_name][np.newaxis, :, :].astype(
                        np.float32),
                    # Add basic attributes for each variable
                    {'grid_mapping': 'spatial_ref'}
                )

            ds_xr = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'time': time_coord,
                    'lat': ('lat', lat_1d, {
                        'standard_name': 'latitude',
                        'long_name': 'Latitude',
                        'units': 'degrees_north',
                        'axis': 'Y'
                    }),
                    'lon': ('lon', lon_1d, {
                        'standard_name': 'longitude',
                        'long_name': 'Longitude',
                        'units': 'degrees_east',
                        'axis': 'X'
                    })
                },
                attrs={
                    'Conventions': 'CF-1.6',
                    'title': 'AMSR3 Resampled Brightness Temperatures'
                }
            )
            ds_xr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds_xr.rio.write_crs("epsg:4326", inplace=True)

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
            filename = os.path.basename(f)
            outfiles.append(full_output_dir / filename)

        return outfiles