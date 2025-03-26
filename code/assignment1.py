import pandas as pd
import numpy as np
import multiprocessing as mp
import timer_wraper as tw
from tqdm import tqdm
import math
import psutil
import matplotlib.pyplot as plt


@tw.timeit
def analyze_large_file_paralelly(file_path, chunk_size=1000, njobs=None, **kwargs):
    if njobs:
        num_workers = njobs
    else:
        num_workers = mp.cpu_count() - 1
    print(f'Using {num_workers} worker processes')
    pool = mp.Pool(processes=num_workers)

    results = []

    total_rows = sum(1 for row in open(file_path)) - 1
    total_chunks = total_rows // chunk_size + 1

    with tqdm(total=total_chunks, unit=' chunk', desc='Processing chunks') as pbar:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):

            chunk['MMSI'] = chunk['MMSI'].astype(str)
            chunk = chunk.loc[ (chunk['Latitude'] <= 90) & (chunk['Longitude'] <= 90) ] # for noise points
            
            mmsis = chunk['MMSI'].unique()
            chunk = chunk.filter(['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG']).drop_duplicates()
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'], dayfirst=True)

            for mmsi in mmsis:
                vessel_data = chunk.loc[chunk['MMSI'] == mmsi].copy()
                result = pool.apply_async(analyze_single_vessel, args=(vessel_data,))
                results.append(result)
                vessel_data = None

            pbar.update(1)
            chunk = None

            if i > 20:
                break


    print("Joining pool...")
    pool.close()
    pool.join()
    collected_results = [r.get() for r in results]
    result = pd.concat(collected_results, ignore_index=True) if collected_results else pd.DataFrame()



    result = result.sort_values(by='speed_delta', na_position='last').drop_duplicates(subset=['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG'], keep='first')
    print("Adding quadrants...")
    result['quadrant'], quadrant_bboxes = add_quadrants(result, **kwargs)

    quadrant_spoofs = result.groupby('quadrant').agg({'spoof': 'mean'}).reset_index()
    quadrant_counts = result['quadrant'].value_counts().reset_index()
    quadrant_counts.columns = ['quadrant', 'count']
    quadrant_spoofs = quadrant_spoofs.merge(quadrant_counts, on='quadrant')
    quadrant_spoofs = quadrant_spoofs.rename(columns={'spoof': 'spoof_rate'})
    quadrant_bboxes = quadrant_bboxes.merge(quadrant_spoofs, left_index=True, right_on='quadrant')
    quadrant_bboxes = quadrant_bboxes.sort_values(by='spoof_rate', ascending=False)


    spoofed_mmsis = result[result['spoof'] == 1]['MMSI'].unique()
    result = result.loc[ result['MMSI'].isin(spoofed_mmsis) ].sort_values(by=['MMSI', '# Timestamp'], ascending=[True, True])
 
    return result, quadrant_bboxes


@tw.timeit
def analyze_large_file_sequentially(file_path, chunk_size=1000, **kwargs):
    print("Sequential mode")
    usage_info = dict(cpu=[], ram=[])

    results = []

    total_rows = sum(1 for row in open(file_path)) - 1
    total_chunks = total_rows // chunk_size + 1

    with tqdm(total=total_chunks, unit=' chunk', desc='Processing chunks') as pbar:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            chunk['MMSI'] = chunk['MMSI'].astype(str)

            chunk = chunk.loc[ (chunk['Latitude'] <= 90) & (chunk['Longitude'] <= 90) ] # for noise points

            mmsis = chunk['MMSI'].unique()
            chunk = chunk.filter(['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG']).drop_duplicates()
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'], dayfirst=True)


            for mmsi in mmsis:
                vessel_data = chunk.loc[chunk['MMSI'] == mmsi].copy()
                result = analyze_single_vessel(vessel_data)
                results.append(result)
                vessel_data = None

            usage_info['ram'].append(psutil.virtual_memory().percent)
            usage_info['cpu'].append(psutil.cpu_percent(percpu=False))

            pbar.update(1)
            chunk = None


    result = pd.concat(results, ignore_index=True)



    # result = result.sort_values(by='speed_delta', na_position='last').drop_duplicates(subset=['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG'], keep='first')
    print("Adding quadrants...")
    result['quadrant'], quadrant_bboxes = add_quadrants(result, **kwargs)

    # quadrant_spoofs = result.groupby('quadrant').agg({'spoof': 'mean'}).reset_index()
    # quadrant_counts = result['quadrant'].value_counts().reset_index()
    # quadrant_counts.columns = ['quadrant', 'count']
    # quadrant_spoofs = quadrant_spoofs.merge(quadrant_counts, on='quadrant')
    # quadrant_spoofs = quadrant_spoofs.rename(columns={'spoof': 'spoof_rate'})
    # quadrant_bboxes = quadrant_bboxes.merge(quadrant_spoofs, left_index=True, right_on='quadrant')
    # quadrant_bboxes = quadrant_bboxes.sort_values(by='spoof_rate', ascending=False)


    # spoofed_mmsis = result[result['spoof'] == 1]['MMSI'].unique()
    # result = result.loc[ result['MMSI'].isin(spoofed_mmsis) ].sort_values(by=['MMSI', '# Timestamp'], ascending=[True, True])
 
    return result, quadrant_bboxes, usage_info


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    if math.isnan(c):
        return 0
    else:
        return R * c


def analyze_single_vessel(vessel_data):
    import warnings 
    warnings.simplefilter("ignore") # for division by zero

    vessel_data.sort_values(by='# Timestamp', ascending=True, inplace=True)

    if len(vessel_data) < 2:
        return pd.DataFrame()
    

    for i in range(len(vessel_data)):
        lat, lon = vessel_data.iloc[i]['Latitude'], vessel_data.iloc[i]['Longitude']
        timestamp = vessel_data.iloc[i]['# Timestamp']

        if i > 0:
            last_timestamp = vessel_data.iloc[i - 1]['# Timestamp']
            last_lat, last_lon = vessel_data.iloc[i - 1]['Latitude'], vessel_data.iloc[i - 1]['Longitude']
            last_speed = vessel_data.loc[vessel_data.index[i - 1], 'speed']

            duration = (timestamp - last_timestamp).total_seconds() / (60*60)
            vessel_data.loc[vessel_data.index[i], 'Time_diff'] = duration

            distance = calculate_distance(last_lat, last_lon, lat, lon)
            vessel_data.loc[vessel_data.index[i], 'distance'] = distance

            speed = np.where((duration > 0) & (distance > 0), distance / duration, 0)
            vessel_data.loc[vessel_data.index[i], 'speed'] = speed

            speed_delta = (speed - last_speed) / last_speed if last_speed > 0 else 1
            speed_delta = 0 if speed == 0 and last_speed == 0 else speed_delta # if both speeds are zero, nothing out of the ordinary
            vessel_data.loc[vessel_data.index[i], 'speed_delta'] = speed_delta

            spoof = int((speed > 60*1.852) or (abs(speed_delta) > 2)) # 60 knots in km/h (a very high speed for a vessel) or more than 2x speed change
            vessel_data.loc[vessel_data.index[i - 1], 'spoof'] = spoof if i > 1 else 0  # only matters where it happened
        else:
            vessel_data.loc[vessel_data.index[i], 'Time_diff'] = 0
            vessel_data.loc[vessel_data.index[i], 'distance'] = 0
            vessel_data.loc[vessel_data.index[i], 'speed'] = 0            
            vessel_data.loc[vessel_data.index[i], 'speed_delta'] = 0
            vessel_data.loc[vessel_data.index[i], 'spoof'] = 0
    
    return vessel_data
   

def quadrant_bounding_boxes(data, num_intervals_lat=4, num_intervals_lon=4):
    quadrant_bounding_boxes = {}

    # Calculate the latitude and longitude step sizes
    lat_step = (data['Latitude'].max() - data['Latitude'].min()) / num_intervals_lat
    lon_step = (data['Longitude'].max() - data['Longitude'].min()) / num_intervals_lon

    # Iterate through each quadrant and calculate the bounding box coordinates
    for i in range(num_intervals_lat):
        for j in range(num_intervals_lon):
            quadrant_id = i * num_intervals_lon + j
            lat_min = data['Latitude'].min() + i * lat_step
            lat_max = lat_min + lat_step
            lon_min = data['Longitude'].min() + j * lon_step
            lon_max = lon_min + lon_step
            quadrant_bounding_boxes[quadrant_id] = {
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max
            }

    return quadrant_bounding_boxes


def add_quadrants(data, **kwargs):
    quadrant_bboxes = quadrant_bounding_boxes(data, **kwargs)
    quadrants = data.apply(
    lambda row: next(
        (qid for qid, bbox in quadrant_bboxes.items()
         if bbox['lat_min'] <= row['Latitude'] <= bbox['lat_max'] and
            bbox['lon_min'] <= row['Longitude'] <= bbox['lon_max']),
        None
    ),
    axis=1
    )
    return quadrants, pd.DataFrame(quadrant_bboxes).transpose()




if __name__ == "__main__":

    (res, quadrants), duration = analyze_large_file_paralelly("/home/viliuskava/studies/Big_data/assignment1/aisdk-2024-10-25.csv", 
                                        20000, 6, num_intervals_lat=8, num_intervals_lon=8)

    res.to_csv("/home/viliuskava/studies/Big_data/assignment1/results/spoofed_mmsis.csv", index=False)
    quadrants.to_csv("/home/viliuskava/studies/Big_data/assignment1/results/spoofed_quadrants.csv", index=False)