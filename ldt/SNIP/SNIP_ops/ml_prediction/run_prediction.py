#!/usr/bin/env python3

# -----------------------BEGIN NOTICE -- DO NOT EDIT-----------------------
# NASA Goddard Space Flight Center
# Land Information System Framework (LISF)
# Version 7.8
# -------------------------END NOTICE -- DO NOT EDIT-----------------------

"""
SCRIPT: run_prediction.py

Script for invoking AI/ML algorithms to retrieve snow depth from AMSR2/3 data.
Uses a Base class to share common ML logic, with specific subclasses for sensors.
"""

from datetime import datetime
import glob
import logging
import os
import time
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from rasterio.enums import Resampling
import xarray as xr
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SnowDepthPredictor')

_ENCODING = {
    'snow_depth': {
        'zlib': True, 'complevel': 4, 'shuffle': True,
        '_FillValue': -9999.0, 'dtype': 'float32'
    },
    'y': {'dtype': 'float32'},
    'x': {'dtype': 'float32'}
}


class BaseSnowDepthPredictor:
    """Base class containing shared ML, data extraction, and formatting logic."""

    def __init__(self, config, sensor_name):
        self.config = config
        self.target_datetime = self.config.target_datetime
        self.sensor_name = sensor_name
        self.model = None
        self.ds_result = None
        self.pmw_file = None

        self.model_feature_names = None
        self.model_channels = [
            '6.9 GHz H', '6.9 GHz V', '7.3 GHz H', '7.3 GHz V',
            '10.7 GHz H', '10.7 GHz V', '18.7 GHz H', '18.7 GHz V',
            '23.8 GHz H', '23.8 GHz V', '36.6 GHz H', '36.6 GHz V',
            '89.0 GHz H', '89.0 GHz V'
        ]

        self.tb_channels = [
            'tb_6h', 'tb_6v', 'tb_7h', 'tb_7v', 'tb_10h', 'tb_10v',
            'tb_18h', 'tb_18v', 'tb_23h', 'tb_23v', 'tb_36h', 'tb_36v',
            'tb_89h', 'tb_89v'
        ]

        self.channel_mapping = dict(zip(self.tb_channels, self.model_channels))

        self.TB_DIFF_PAIRS = [
            ('10.7 GHz V', '18.7 GHz V', '10.7_V-18.7_V'),
            ('10.7 GHz V', '23.8 GHz V', '10.7_V-23.8_V'),
            ('10.7 GHz V', '36.6 GHz V', '10.7_V-36.6_V'),
            ('18.7 GHz V', '23.8 GHz V', '18.7_V-23.8_V'),
            ('18.7 GHz V', '36.6 GHz V', '18.7_V-36.6_V'),
            ('23.8 GHz V', '36.6 GHz V', '23.8_V-36.6_V'),
        ]

    def load_model(self) -> None:
        try:
            start_time = time.time()
            model_path = (self.config.project_path /
                          f"{self.config.model_path}_{self.sensor_name}.json")
            #-------------------
            # TODO remove this chunk once AMSR3 ML model is finalized
            if self.sensor_name == "AMSR3":
                model_path = (self.config.project_path /
                              f"{self.config.model_path}_AMSR2.json")
            # -------------------
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.model = model
            self.model_feature_names = model.feature_names_in_.tolist()
            logger.info(
                "%s Model loaded in %.2f seconds",
                self.sensor_name,
                time.time() - start_time,
            )
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def _add_tb_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        for c1, c2, name in self.TB_DIFF_PAIRS:
            if c1 in df.columns and c2 in df.columns:
                df[name] = df[c1] - df[c2]
        return df

    def add_days_since_wy(self, df):
        df = df.copy()
        date_format = "%Y%m%d%H%M" if len(
            str(df['Date'].iloc[0])) > 10 else "%Y%m%d%H"
        df['date'] = pd.to_datetime(df['Date'], format=date_format)

        nh_mask = df['lat'] >= 0
        nh_oct_plus = (df['date'].dt.month >= 10) & nh_mask
        nh_jan_sep = (df['date'].dt.month < 10) & nh_mask

        sh_mask = df['lat'] < 0
        sh_apr_plus = (df['date'].dt.month >= 4) & sh_mask
        sh_jan_mar = (df['date'].dt.month < 4) & sh_mask

        wy_starts = pd.Series(index=df.index, dtype='datetime64[ns]')
        wy_starts[nh_oct_plus] = pd.to_datetime(
            df.loc[nh_oct_plus, 'date'].dt.year.astype(str) + '-10-01')
        wy_starts[nh_jan_sep] = pd.to_datetime(
            (df.loc[nh_jan_sep, 'date'].dt.year - 1).astype(str) + '-10-01')
        wy_starts[sh_apr_plus] = pd.to_datetime(
            df.loc[sh_apr_plus, 'date'].dt.year.astype(str) + '-04-01')
        wy_starts[sh_jan_mar] = pd.to_datetime(
            (df.loc[sh_jan_mar, 'date'].dt.year - 1).astype(str) + '-04-01')

        df['days_since_wy'] = (df['date'] - wy_starts).dt.days
        df['sin_period'] = np.sin(2 * np.pi * df['days_since_wy'] / 365.25)
        df['cos_period'] = np.cos(2 * np.pi * df['days_since_wy'] / 365.25)
        return df

    def _extract_features(self, data: xr.Dataset,
                          target_datetime: str) -> pd.DataFrame:
        df = pd.DataFrame()

        # 1. Extract TB Channels
        for tb_channel in self.tb_channels:
            model_channel = self.channel_mapping[tb_channel]
            if hasattr(data, tb_channel):
                tb_data = getattr(data, tb_channel).squeeze()
                df[model_channel] = tb_data.values.flatten()

        df['Date'] = target_datetime

        # 2. Extract Coordinates SAFELY
        lat_vals = data['lat'].squeeze().values
        lon_vals = data['lon'].squeeze().values

        # If they are 1D, mesh them together into a 2D grid
        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
            Lon, Lat = np.meshgrid(lon_vals, lat_vals)
        else:
            # If they are already 2D (1920, 2560), just use them directly
            Lon, Lat = lon_vals, lat_vals

        df['lat'] = Lat.flatten()
        df['lon'] = Lon.flatten()

        # 3. Add Time and Difference Features
        df = self.add_days_since_wy(df)
        df = self._add_tb_differences(df)

        return df

    def _apply_model(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        self.load_model()
        df = df[self.model_feature_names]
        nan_mask = np.isnan(df).any(axis=1)
        x_test_no_nan = df[~nan_mask]

        if len(x_test_no_nan) < 100:
            return None

        y_pred_no_nan = self.model.predict(x_test_no_nan)
        y_pred = np.full(df.shape[0], np.nan)
        y_pred[~nan_mask] = y_pred_no_nan
        y_pred[y_pred <= 0.001] = 0
        return y_pred

    def _format_output(self, predictions: np.ndarray, shape: Tuple,
                       y_coords: np.ndarray,
                       x_coords: np.ndarray) -> xr.DataArray:

        reshaped_predictions = np.where(predictions.reshape(shape) < 0, np.nan,
                                        predictions.reshape(shape))

        output = xr.DataArray(
            reshaped_predictions,
            coords={
                'y': ('y', y_coords,
                      {'long_name': 'Latitude', 'standard_name': 'latitude',
                       'units': 'degrees_north'}),
                'x': ('x', x_coords,
                      {'long_name': 'Longitude', 'standard_name': 'longitude',
                       'units': 'degrees_east'})
            },
            dims=["y", "x"],
            attrs={'long_name': 'Snow Depth', 'units': 'meters',
                   'valid_range': [0.0, 10.0]}
        )
        return output.rio.write_crs(self.config.proj, inplace=True)

    def _process_pmw_data(self, pmw_file: str, target_datetime: str) -> \
    Optional[xr.DataArray]:
        ds_pmw = xr.open_dataset(pmw_file, decode_timedelta=False)
        try:
            df = self._extract_features(ds_pmw, target_datetime)
            y_pred = self._apply_model(df)
            if y_pred is None: return None

            lat = ds_pmw.squeeze().lat.values
            lon = ds_pmw.squeeze().lon.values
            return self._format_output(y_pred, (len(lat), len(lon)), lat, lon)
        finally:
            ds_pmw.close()

    def predict_snow_depth(self, pmw_file) -> Optional[xr.DataArray]:
        output_file, target_datetime = self.get_file_paths(pmw_file)
        if os.path.exists(output_file): return None
        return self._process_pmw_data(pmw_file, target_datetime)

    def apply_filter(self):
        data = self.ds_result
        is_coldsnow = (data['tb_36h'] < 245) & (data['tb_36v'] < 255)
        is_medium_deep = (data['tb_10v'] > data['tb_36v']) | (
                    data['tb_10h'] > data['tb_36h'])
        T = 58.08 - 0.39 * data['tb_18v'] + 1.21 * data['tb_23v'] - 0.37 * data[
            'tb_36h'] + 0.36 * data['tb_89v']

        is_shallow = ((data['tb_89v'] <= 255) & (data['tb_89h'] <= 265) &
                      (data['tb_23v'] > data['tb_89v']) & (
                                  data['tb_23h'] > data['tb_89h']) & (T < 267))

        depth_flag = np.full_like(data['tb_36h'].values, np.nan, dtype=float)
        valid_mask = ~np.isnan(data['tb_36h'].values)
        depth_flag[valid_mask] = 0
        depth_flag = np.where(is_shallow.values & valid_mask, 2, depth_flag)
        depth_flag = np.where(is_medium_deep.values & valid_mask, 1, depth_flag)
        depth_flag = np.where(is_coldsnow.values & valid_mask, 1, depth_flag)

        data['depth_flag'] = (data['tb_36h'].dims, depth_flag)
        self.ds_result = data
        return data

    def apply_snow_mask(self, ds_out, template_data):
        date_str = self.target_datetime.strftime('%Y%m%d')
        viirs_path = glob.glob(
            os.path.join(self.config.viirs_path, f'snomap*{date_str}*tiff'))
        if len(viirs_path) == 1:
            ds_viirs = xr.open_dataset(viirs_path[0], engine='rasterio',
                                       decode_timedelta=False)
            ds_viirs = ds_viirs.rio.write_crs(self.config.proj).squeeze()
            ds_viirs_repro = ds_viirs.rio.reproject_match(template_data,
                                                          resampling=Resampling.mode)
            ds_out['snow_depth'] = xr.where(ds_viirs_repro['band_data'] <= 0.5,
                                            0, ds_out['snow_depth'])
        return ds_out


class AMSR2SnowDepthPredictor(BaseSnowDepthPredictor):
    """AMSR2 Specific Implementation (Single file processing, bit-masking)"""

    def __init__(self, config=None):
        super().__init__(config, "AMSR2")

    def get_file_paths(self, pmw_file=None) -> Tuple[str, str]:
        target_datetime = self.target_datetime.strftime("%Y%m%d%H")
        dir_out = self.config.project_path / self.config.output_dir
        os.makedirs(dir_out, exist_ok=True)
        return os.path.join(dir_out,
                            f'amsr2_snip_0p1deg_{target_datetime}.nc'), target_datetime

    def save_to_netcdf(self, output: xr.DataArray, pmw_file: str) -> bool:
        output_file, _ = self.get_file_paths()
        ds_out = output.to_dataset(name="snow_depth")

        self.ds_result = xr.open_dataset(pmw_file,
                                         decode_timedelta=False).squeeze()

        if self.config.land_frac_th is not None:
            land_mask = self.ds_result['land_percent'].squeeze()
            if land_mask.dims == ('lat', 'lon') or land_mask.dims == ['lat',
                                                                      'lon']:
                land_mask = land_mask.rename({'lat': 'y', 'lon': 'x'})
            land_mask = land_mask >= self.config.land_frac_th
            ds_out['snow_depth'] = ds_out['snow_depth'].where(land_mask)

        if getattr(self.config, 'flag_cold', False):
            flag_values_to_mask = [1, 2, 3, 5]
            if getattr(self.config, 'flag_rain',
                       False): flag_values_to_mask.append(0)
            if getattr(self.config, 'flag_rfi',
                       False): flag_values_to_mask.extend([6, 7])

            flag_cleaned = self.ds_result['pixel_qual_flag'].fillna(255)
            flag_uint8 = flag_cleaned.astype('uint8')
            flag_mask = xr.zeros_like(flag_uint8, dtype=bool)

            for bit in flag_values_to_mask:
                flag_mask = flag_mask | ((flag_uint8 & (1 << bit)) != 0)

            flag_mask = flag_mask.squeeze()
            if flag_mask.dims == ('lat', 'lon'): flag_mask = flag_mask.rename(
                {'lat': 'y', 'lon': 'x'})
            ds_out = ds_out.where(~flag_mask.squeeze())

        if getattr(self.config, 'flag_shallow', False):
            ds_pmw_shallow = self.apply_filter()
            snow_depth_arr = ds_out['snow_depth'].values.copy()
            snow_depth_arr[ds_pmw_shallow['depth_flag'].values == 0] = 0
            snow_depth_arr[ds_pmw_shallow['depth_flag'].values == 2] = 0.05
            ds_out['snow_depth'].values = snow_depth_arr

        ds_out.to_netcdf(output_file, encoding=_ENCODING, format='NETCDF4')
        return True

    def run_pipeline(self, pmw_file) -> bool:
        output_file, _ = self.get_file_paths()
        if os.path.exists(output_file): return True

        output = self.predict_snow_depth(pmw_file=pmw_file)
        if output is None: return False
        return self.save_to_netcdf(output=output, pmw_file=pmw_file)


class AMSR3SnowDepthPredictor(BaseSnowDepthPredictor):
    """AMSR3 Specific Implementation (Multiple swaths, 6-hr merge,
    simplified masking)"""

    def __init__(self, config=None):
        super().__init__(config, "AMSR3")

    def get_file_paths(self, pmw_file=None) -> Tuple[str, str]:
        pred_dir = self.config.project_path / getattr(
            self.config, 'amsr3_ml_sd_path', './data/amsr3_ml_sd')
        os.makedirs(pred_dir, exist_ok=True)

        if pmw_file:
            basename = os.path.basename(str(pmw_file))
            target_str = self.target_datetime.strftime("%Y%m%d%H00")
            return os.path.join(pred_dir, basename), target_str

        else:
            target_str = self.target_datetime.strftime("%Y%m%d%H")
            return os.path.join(pred_dir,
                                f'AMSR3_snip_0p1deg_{target_str}.nc'), target_str

    def save_to_netcdf(self, output: xr.DataArray, pmw_file: str) -> bool:
        output_file, _ = self.get_file_paths(pmw_file)

        # Convert to Dataset
        ds_out = output.to_dataset(name="snow_depth")

        # 1. Update Coordinates for CF Compliance
        ds_out.coords['y'].attrs.update({
            'long_name': 'Latitude', 'standard_name': 'latitude',
            'units': 'degrees_north', 'valid_range': [-90., 90.],
            'axis': 'Y'
        })
        ds_out.coords['x'].attrs.update({
            'long_name': 'Longitude', 'standard_name': 'longitude',
            'units': 'degrees_east', 'valid_range': [-180., 180.],
            'axis': 'X'
        })

        # 2. Add Global Attributes
        ds_out.attrs.update({
            'title': 'AMSR3 Snow Depth Prediction',
            'Conventions': 'CF-1.8',
            'source': 'XGBoost model prediction'
        })

        # 3. Add Variable Attributes
        ds_out['snow_depth'].attrs.update({
            'units': 'meters',
            'long_name': 'Snow Depth',
            'standard_name': 'surface_snow_thickness',
            'valid_range': [0.0, 10.0],
            'grid_mapping': 'spatial_ref'
        })

        # 4. Embed the actual Projection Math (EPSG:4326 WGS84)
        ds_out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        ds_out.rio.write_crs("epsg:4326", inplace=True)

        # ----------------------------------------------------
        # Masking and filtering logic (Same as before)
        self.ds_result = xr.open_dataset(pmw_file,
                                         decode_timedelta=False).squeeze()

        if self.config.land_frac_th is not None:
            land_frac = self.ds_result['land_percent'].squeeze()
            if land_frac.dims == ('lat', 'lon') or land_frac.dims == ['lat',
                                                                      'lon']:
                land_frac = land_frac.rename({'lat': 'y', 'lon': 'x'})
            land_mask = land_frac >= self.config.land_frac_th
            ds_out['snow_depth'] = ds_out['snow_depth'].where(land_mask)

        if getattr(self.config, 'flag_shallow', False):
            ds_pmw_shallow = self.apply_filter()
            snow_depth_arr = ds_out['snow_depth'].values.copy()
            depth_flag_arr = ds_pmw_shallow['depth_flag'].values

            snow_depth_arr[depth_flag_arr == 0] = 0
            snow_depth_arr[depth_flag_arr == 2] = 0.05
            ds_out['snow_depth'].values = snow_depth_arr

        # Save to disk
        ds_out.to_netcdf(output_file, encoding=_ENCODING, format='NETCDF4')
        return True

    def merge_6hr_outputs(self, output_files: list,
                          final_output_file: str) -> bool:
        ds_list = []
        try:
            for f in output_files:
                ds = xr.open_dataset(f, decode_timedelta=False)
                if 'snow_depth' not in ds:
                    raise KeyError(f"{f} does not contain 'snow_depth'")
                ds_list.append(ds)

            ds_merged = xr.concat(ds_list, dim='time').mean(dim='time',
                                                            skipna=True)
            if 'snow_depth' not in ds_merged:
                raise KeyError("Merged dataset does not contain 'snow_depth'")

            ds_merged = ds_merged.expand_dims(
                time=pd.to_datetime([self.target_datetime]))

            ds_merged.attrs = ds_list[0].attrs
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            ds_merged.to_netcdf(final_output_file, encoding=_ENCODING,
                                format='NETCDF4')

            for f in output_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        logger.info("Removed temporary AMSR3 SD file after merge: %s",
                                    f)
                    except OSError as exc:
                        logger.warning("Could not remove %s: %s", f, exc)

            return True
        except Exception as e:
            logger.error("Error merging AMSR3 files: %s", e)
            return False
        finally:
            for ds in ds_list:
                ds.close()

    def run_pipeline(self, pmw_files: Union[str, List[str]]) -> bool:
        if not isinstance(pmw_files, list):
            pmw_files = [pmw_files]
        pmw_files = [str(f) for f in pmw_files]

        final_output_file, _ = self.get_file_paths()
        if os.path.exists(final_output_file): return True

        output_files_generated = []
        for pmw_file in pmw_files:
            out_file, _ = self.get_file_paths(pmw_file)
            if os.path.exists(out_file):
                output_files_generated.append(out_file)
                continue

            output = self.predict_snow_depth(pmw_file=pmw_file)
            if output is not None and self.save_to_netcdf(output=output,
                                                          pmw_file=pmw_file):
                output_files_generated.append(out_file)

        if output_files_generated:
            return self.merge_6hr_outputs(output_files_generated,
                                          final_output_file)
        return False