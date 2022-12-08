"""
This takes all the *.uf files in:
/home/rolan/Weather-Datasets/weather_data/weather_data/batch1/Radar/2019/Subic/subic_p
and converts them to numpy files, to be put in:
home/rolan/Weather-Datasets/npy-all-sweeps-max

The conversion takes all the original .uf files' sweeps and takes the maxima across all sweeps, and saves these as a numpy file.
"""
import pyart
from pyart.core import Radar
import os
import numpy as np
from numpy.ma import MaskedArray
source_folder = "/home/rolan/Weather-Datasets/weather_data/weather_data/batch1/Radar/2019/Subic/subic_p"
destination_folder = "/home/rolan/Weather-Datasets/npy-all-sweeps-max"


def transform_to_grid(radar: Radar, sweep_idx: int) -> MaskedArray:
    """
    This takes a pyart.core.Radar object, and the sweep index, and
    :param radar:
    :param sweep_idx:
    :return:
    """
    sweep = radar.extract_sweeps([0])
    sweep_data = radar.extract_sweeps([sweep_idx]).fields['corrected_reflectivity']['data']
    sweep_azi = radar.extract_sweeps([sweep_idx]).azimuth['data']
    sweep.fields['corrected_reflectivity']['data'] = sweep_data
    sweep.azimuth['data'] = sweep_azi

    grid = pyart.map.grid_from_radars(sweep, grid_shape=(1, 256, 256), grid_limits=((0, 2000), (-128000, 128000), (-128000, 128000)),
                                      fields=['corrected_reflectivity'])

    return grid.fields['corrected_reflectivity']['data']


def process_uf_file(source_file: str, dest_file: str):
    radar = pyart.io.read_uf(source_file)
    max_arr = np.zeros((256, 256))
    for s in range(radar.nsweeps):
        sweep = transform_to_grid(radar=radar, sweep_idx=s)
        max_arr = np.maximum(max_arr, sweep)

    #np.save(dest_file, max_arr)
    # MaskedArray.tofile() is not yet implemented.  https://stackoverflow.com/questions/13877063/how-to-save-numpy-masked-array-to-file suggests
    # doing the following:
    np.savez_compressed(dest_file, data=max_arr.data, mask=max_arr.mask)

    # the same post suggested loading it as:
    # with np.load(dest_file) as npz:
    #      max_arr = np.ma.MaskedArray(**npz)


def main():
    # The files are in source_folder/month/day/file.uf. We need to iterate through all months, and all days in each month.
    month_folders = sorted([m for m in os.listdir(source_folder)])
    for m in month_folders:  # m is '01', '02',...., '09'
        target_month_folder = os.path.join(destination_folder, m)
        if not os.path.exists(target_month_folder):
            os.makedirs(target_month_folder)
        day_folders = sorted([d for d in os.listdir(os.path.join(source_folder, m))])
        for d in day_folders:  # d is '01',.....,'31'
            print(f"Processing month: {m} day: {d}")
            target_day_folder = os.path.join(destination_folder, m, d)
            if not os.path.exists(target_day_folder):
                os.makedirs(target_day_folder)
            uf_files = sorted([f for f in os.listdir(os.path.join(source_folder, m, d)) if f.endswith('.uf')])
            for f in uf_files:
                uf_file = os.path.join(source_folder, m, d, f)
                dest_filename = f.split('.')[0] + '.npz'
                target_file = os.path.join(destination_folder, m, d, dest_filename)
                process_uf_file(source_file=uf_file, dest_file=target_file)


if __name__ == "__main__":
    main()
