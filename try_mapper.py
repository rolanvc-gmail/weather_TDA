import kmapper as km
import numpy as np
import os
import glob
from numpy import ndarray
from sklearn import manifold
import time


def get_flat_data(month: int, day: int, size:int) -> ndarray:
    """
    This gets radar data from the data_folder specified below, and stacks these into an ndarray, then returns the result
    :param month: the month to get the data from
    :param day: the day of the month to get the data from
    :param size: how many files of data to get. Each file is data taken from 10 minute intervals.
    :return: an ndarray of the list of data.
    """
    result = []
    data_folder = f"/home/rolan/data1/Weather-Datasets/npy-data/{month:02}/{day:02}"
    files = glob.glob(os.path.join(data_folder, "*"))
    for count, file in enumerate(files):
        if count >= size:
            break  # since we keep crashing when at 111 days, let's start with a small number first.
        file_data = np.load(file)
        flat_data = file_data.flatten()
        result.append(flat_data)
    return np.asarray(result)


def do_mapper(data: ndarray, desc: str):
    # Create KeplerMapper object.
    mapper = km.KeplerMapper(verbose=1)

    # Compute projected_data using mapper.fit_transform
    print(f"Compute projected data using mapper.fit_transform")
    projected_data = mapper.fit_transform(data, projection=manifold.TSNE)
    print(f"projected_data has shape: {projected_data.shape}")


    #Create a cover
    cover = km.Cover(n_cubes=10, perc_overlap=0.5, limits=None, verbose=0)

    # Run the mapper algorithm
    print(f"Running the mapper algorithm")
    start = time.time()
    graph = mapper.map(projected_data, data, cover=cover)
    end = time.time()
    print(f"mapper took {(end-start)/60} minutes.")

    output_path = "output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Visualize and save html
    filename = f"kmapper_weather_{desc}"
    _ = mapper.visualize(graph, path_html=os.path.join(output_path, filename), title=desc)


def main():
    day = 9
    month = 1
    size = 10
    data = get_flat_data(month=month, day=day, size=size)
    print(f"data has shape: {data.shape}")
    desc = f"Subic_p_{month}_{day}_size_{size}"
    do_mapper(data, desc)



if __name__ == "__main__":
    main()












