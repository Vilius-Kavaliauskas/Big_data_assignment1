import pandas as pd
import numpy as np
import multiprocessing as mp
import timer_wraper as tw
from tqdm import tqdm
import math
import psutil
import matplotlib.pyplot as plt


@tw.timeit
def analyze_large_file(file_path, chunk_size=1000, mode='parallel', njobs=None, measure_per_cpu=False, **kwargs):
    if mode == 'parallel':
        if njobs:
            num_workers = njobs
        else:
            num_workers = mp.cpu_count() - 1
        print(f'Using {num_workers} worker processes')
        pool = mp.Pool(processes=num_workers)
        manager = mp.Manager()
        shared_vessel_info = manager.dict()
        usage_info = dict(percpu=[], cpu=[], ram=[])



    else:
        print("Sequential mode")
        shared_vessel_info = {}
        usage_info = dict(cpu=[], ram=[])


    results = []

    total_rows = sum(1 for row in open(file_path)) - 1
    total_chunks = total_rows // chunk_size + 1

    with tqdm(total=total_chunks, unit=' chunk', desc='Processing chunks') as pbar:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):

            chunk['MMSI'] = chunk['MMSI'].astype(str)
            mmsis = chunk['MMSI'].unique()

            chunk = chunk.filter(['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG']).drop_duplicates()
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'], dayfirst=True)


            for mmsi in mmsis:
                vessel_data = chunk.loc[chunk['MMSI'] == mmsi].copy()

                usage_info['ram'].append(psutil.virtual_memory().percent)

                if mode == 'parallel':
                    result = pool.apply_async(analyze_single_vessel, args=(vessel_data, shared_vessel_info, usage_info,))
                    if measure_per_cpu:
                        usage_info['percpu'].append(psutil.cpu_percent(percpu=True))

                elif mode == 'sequential':
                    result = analyze_single_vessel(vessel_data, shared_vessel_info, usage_info)
                
                usage_info['cpu'].append(psutil.cpu_percent(percpu=False))
                
                results.append(result)

            pbar.update(1)

            if i >= 20:
                break


    if mode == 'parallel':
        pool.close()
        pool.join()
        collected_results = [r.get() for r in results]
        result = pd.concat(collected_results, ignore_index=True) if collected_results else pd.DataFrame()


    elif mode == 'sequential':
        result = pd.concat(results, ignore_index=True)


    result = result.sort_values(by='speed_delta', na_position='last').drop_duplicates(subset=['# Timestamp', 'Navigational status', 'MMSI', 'Latitude', 'Longitude', 'COG'], keep='first')

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


def analyze_single_vessel(chunk, shared_vessel_info, usage_info):
    import warnings 
    warnings.simplefilter("ignore") # for division by zero

    vessel_data = chunk.copy()
    vessel_data.sort_values(by='# Timestamp', ascending=True, inplace=True)

    if len(vessel_data) < 2:
        return pd.DataFrame()
    
    mmsi = vessel_data['MMSI'].iloc[0]
    
    if mmsi in shared_vessel_info: # look for the last info about this vessel in shared memory
        last_info = shared_vessel_info[mmsi]
        last_lat, last_lon, last_speed, last_timestamp = last_info['lat'], last_info['lon'], last_info['speed'], last_info['timestamp']
    else:
        last_lat, last_lon, last_speed, last_timestamp = None, None, None, None

    for i in range(len(vessel_data)):
        lat, lon = vessel_data.iloc[i]['Latitude'], vessel_data.iloc[i]['Longitude']
        timestamp = vessel_data.iloc[i]['# Timestamp']

        if last_lat is not None and last_lon is not None:
            duration = (timestamp - last_timestamp).total_seconds() / (60*60)
            vessel_data.loc[vessel_data.index[i], 'Time_diff'] = duration

            distance = calculate_distance(last_lat, last_lon, lat, lon)
            vessel_data.loc[vessel_data.index[i], 'distance'] = distance

            speed = np.where((duration > 0) & (distance > 0), distance / duration, 0)
            vessel_data.loc[vessel_data.index[i], 'speed'] = speed

            speed_delta = (speed - last_speed) / last_speed if last_speed > 0 else 1
            vessel_data.loc[vessel_data.index[i], 'speed_delta'] = speed_delta

            spoof = int((speed > 60*1.852) or (abs(speed_delta) > 0.5)) # 60 knots in km/h (a very high speed for a vessel) or more than 50% speed change
            vessel_data.loc[vessel_data.index[i - 1], 'spoof'] = spoof if i > 1 else 0 # only matters where it happened
        else:
            vessel_data.loc[vessel_data.index[i], 'Time_diff'] = 0
            vessel_data.loc[vessel_data.index[i], 'distance'] = 0
            vessel_data.loc[vessel_data.index[i], 'speed'] = 0            
            vessel_data.loc[vessel_data.index[i], 'speed_delta'] = 0
            vessel_data.loc[vessel_data.index[i], 'spoof'] = 0

        last_lat, last_lon, last_speed, last_timestamp = lat, lon, vessel_data.loc[vessel_data.index[i], 'speed'], timestamp

    # usage_info['cpu'].append(psutil.cpu_percent())
    # usage_info['ram'].append(psutil.virtual_memory().percent)

    shared_vessel_info[mmsi] = {'lat': last_lat, 'lon': last_lon, 'speed': last_speed, 'timestamp': last_timestamp} # save the last info for the next chunk
    

    return vessel_data
    

def quadrant_bounding_boxes(data, num_intervals_lat=4, num_intervals_lon=4):
    # Initialize a dictionary to store the bounding box coordinates of each quadrant
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

    num_workers = np.arange(6, 8)

    # percpu_usages = []
    total_cpu_usages = []
    ram_usages = []
    durations = []

    for njobs in num_workers:
        (res, quadrants, usage), duration = analyze_large_file("/home/viliuskava/studies/Big_data/assignment1/aisdk-2024-10-25.csv", 
                                        1000, "parallel", njobs, False, num_intervals_lat=8, num_intervals_lon=8)
        # cpu_usage_per_unit = sum([ sum(usage) for usage in usage['percpu'] ]) / len(usage['percpu'])
        total_cpu_usage = sum(usage['cpu']) / len(usage['cpu'])
        ram_usage = sum(usage['ram']) / len(usage['ram'])
        # percpu_usages.append(cpu_usage_per_unit)
        total_cpu_usages.append(total_cpu_usage)
        ram_usages.append(ram_usage)
        durations.append(duration)


    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # axs[0].plot(num_workers, percpu_usages, label='CPU usage per unit')
    axs[0].plot(num_workers, total_cpu_usages, label='Total CPU usage')
    axs[0].plot(num_workers, ram_usages, label='RAM usage')
    axs[0].set_xlabel('Number of workers')
    axs[0].set_ylabel('Usage (%)')
    axs[0].legend()
    axs[0].set_title('Resource usage')


    axs[1].plot(num_workers, durations, label='Duration', color='red')
    axs[1].set_ylabel('Duration (s)')
    axs[1].set_xlabel('Number of workers')
    axs[1].set_title('Execution duration')

    plt.savefig("/home/viliuskava/studies/Big_data/assignment1/results/results.png")





    # print("==========================================")
    # (res2, quadrants2, usage2), duration_sequential = analyze_large_file("C:\\Users\\kavav\\Documents\\1 kursas (mag)\\2 semestras\\Big data\\downloads\\aisdk-2024-10-25.csv", 
    #                                     1000, "sequential", num_intervals_lat=8, num_intervals_lon=8)
    # print("-----------------------------")

    # total_cpu_usage2 = sum(usage2['cpu']) / len(usage2['cpu'])
    # ram_usage2 = sum(usage2['ram']) / len(usage2['ram'])
    # print("Average CPU usage:", round(total_cpu_usage2, 3))
    # print("Average RAM usage:", round(ram_usage2, 3))
    # print("Sequential execution duration:", round(duration_sequential, 3))

    # print("==========================================")
    # print("Datasets equal:", res.equals(res2))

