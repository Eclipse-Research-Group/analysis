import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr
import matplotlib.dates as mdates
from datetime import datetime, timezone
import scipy
import re
import os

# Compile the regular expression for better performance if checking multiple lines
METADATA_PATTERN = re.compile(r'^# ([A-Z_]+)\t+(\S+)$')

def read_metadata(filename):
    metadata = {}
    parsing = False
    with open(filename, "r") as f:
        for line in f:
            if parsing:
                if line == "## END METADATA ##\n":
                    break
                else:
                    # Use re.match to check if the line matches the pattern and extract groups if it does
                    match = METADATA_PATTERN.match(line)
                    if match:
                        # Extracting the groups as (key, value)
                        key, value = match.groups()
                        metadata[key] = value
                    else:
                        print("Malformed metadata line: ", line)
                
            if line == "## BEGIN METADATA ##\n":
                parsing = True
    
    return metadata

def parse_line_v3(line):
    try:
        parts = line.strip().split(",")
        # Unpacking the fixed parts
        (computer_time_str, gps_time_str, flags, sample_rate_str, latitude_str, longitude_str, 
         elevation_str, satellite_count_str, speed_str, heading_str, count_samples_str, *samples_with_checksum) = parts
        
        # Converting strings to appropriate types
        computer_time = pd.to_datetime(float(computer_time_str), unit="s")
        gps_time = pd.to_datetime(float(gps_time_str), unit="s")
        sample_rate = float(sample_rate_str)
        latitude = float(latitude_str)
        longitude = float(longitude_str)
        elevation = float(elevation_str)
        satellite_count = int(satellite_count_str)
        speed = float(speed_str)
        heading = float(heading_str)
        count_samples = int(count_samples_str)
        
        # Processing samples and checksum from the remaining list
        samples = np.array([int(sample) for sample in samples_with_checksum[:count_samples]], dtype=np.uint16)
        # samples = (samples - 512) / 512
        checksum = int(samples_with_checksum[count_samples])  # Assuming checksum is right after samples

        return {
            "computer_time": computer_time,
            "gps_time": gps_time,
            "flags": flags,
            "sample_rate": sample_rate,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "satellite_count": satellite_count,
            "speed": speed,
            "heading": heading,
            "count_samples": count_samples,
            "samples": samples,
            "checksum": checksum,
        }
    except Exception as e:
        raise e

def parse_line_v2(line):
    try:
        parts = line.strip().split(",")
        # Unpacking the fixed parts
        (gps_time_str, flags, sample_rate_str, latitude_str, longitude_str, 
         elevation_str, satellite_count_str, speed_str, heading_str, count_samples_str, *samples_with_checksum) = parts
        
        # Converting strings to appropriate types
        gps_time = pd.to_datetime(float(gps_time_str), unit="s")
        sample_rate = float(sample_rate_str)
        latitude = float(latitude_str)
        longitude = float(longitude_str)
        elevation = float(elevation_str)
        satellite_count = int(satellite_count_str)
        speed = float(speed_str)
        heading = float(heading_str)
        count_samples = int(count_samples_str)
        
        # Processing samples and checksum from the remaining list
        samples = np.array([int(sample) for sample in samples_with_checksum[:count_samples]], dtype=np.uint16)
        # samples = (samples - 512) / 512
        checksum = int(samples_with_checksum[count_samples])  # Assuming checksum is right after samples

        return {
            "gps_time": gps_time,
            "flags": flags,
            "sample_rate": sample_rate,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "satellite_count": satellite_count,
            "speed": speed,
            "heading": heading,
            "count_samples": count_samples,
            "samples": samples,
            "checksum": checksum,
        }
    except Exception as e:
        raise e

def parse_line_v1(line):
    try:
        parts = line.strip().split(",")
        # Unpacking the fixed parts
        (time_str, *samples) = parts
        
        # Converting strings to appropriate types
        time = pd.to_datetime(float(time_str), unit="s")
        
        # Processing samples and checksum from the remaining list
        samples = np.array([int(sample) for sample in samples], dtype=np.uint16)
        # samples = (samples - 512) / 512

        if samples.shape[0] != 512 and samples.shape[0] != 1024:
            raise Exception(f"Bad line: {samples.shape}")

        return {
            "computer_time": time,
            "samples": samples
        }
    except Exception as e:
        raise e


def load_file(filename, sample_rate=20000, debug=False):
    metadata = read_metadata(filename)
    version = 2
    errors = 0
    parse_function = parse_line_v2
    time_field = "gps_time"
    attrs = {}
    
    if "VERSION" in metadata:
        version = int(metadata["VERSION"])
        parse_function = parse_line_v3
        time_field = "computer_time"
    elif metadata == {}:
        version = 1
        parse_function = parse_line_v1
        time_field = "computer_time"

    if version > 3:
        raise Exception(f"Version {version} not supported!")

    if version == 2 or version == 3:
        attrs["capture_id"] = metadata["CAPTURE_ID"]
        attrs["node_id"] = metadata["NODE_ID"]
        attrs["created"] = metadata["CREATED"]
        attrs["sample_rate"] = int(metadata["SAMPLE_RATE"])
    else:
        attrs["sample_rate"] = sample_rate

    attrs["schema_version"] = version

    if debug:
        print(f"Detected version {version}")

    
    count = 0
    with open(filename, "r") as f:
        data_entries = []
        current_time_counter = 0
        if debug:
            print("Reading file")
        for line in f:
            if count > 2:
                break
            
            if line.startswith(("#", "$")):
                continue
    
            try:
                parsed_data = parse_function(line)
                data_entries.append(parsed_data)
                count += 1
            except Exception as e:
                if debug:
                    print(f"Error parsing line: {line[:30]}: {e}")
                errors += 1
                continue

    if len(data_entries) == 0:
        return xr.Dataset()
    
    # Prepare data for xarray
    times = [entry[time_field] for entry in data_entries]
    samples = np.array([entry["samples"] for entry in data_entries])

    # for entry in data_entries:
    #     samples = entry["samples"].shape[0]
    #     the_length = scipy.stats.mode(samples.shape[0])
    
    
    # Additional data variables
    data_vars = {key: ("time", [entry[key] for entry in data_entries]) for key in data_entries[0] if key not in [time_field, "samples"]}
    data_vars["samples"] = (("time", "sample"), samples)
    
    # Coordinates
    coords = {
        "time": times,
        "sample": np.arange(samples.shape[1])
    }
    
    # Create the xarray dataset
    xr_dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    if debug:
        print(f"File successfully parsed, {errors} errors occurred.")
        
    return xr_dataset



def generate_heatmap(dataset, variable_name, time_name, colormap='seismic', xlim=(0,0.03), figsize=(12, 8), dpi=200, skip=1, light_mode=False):
    # Settings for plotting
    sample_rate = dataset.attrs["sample_rate"]
    count = dataset["samples"].shape[1]
    end_time_seconds = count / sample_rate
    
    if not light_mode:
        plt.style.use('dark_background')
        
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Generate x and y ranges
    x_range = np.linspace(0, end_time_seconds * 1000, count) # Converting seconds to milliseconds
    
    # Convert gps_time to matplotlib date format for plotting
    y_range_mpl = mdates.date2num(pd.to_datetime(dataset[time_name].values))
    
    # Assuming the "samples" variable is what you want to plot and has the shape (time, sample)
    # Let's reshape it to a 2D array if it's not already
    data_2d = dataset[variable_name].values
    
    # The data for pcolormesh needs to be transposed if the time dimension is first
    data_2d_transposed = data_2d.T
    
    # Plotting
    X, Y = np.meshgrid(x_range, y_range_mpl)
    im = ax.pcolormesh(X, Y, data_2d, cmap=colormap, shading='nearest', vmin=-1, vmax=1)
    
    # Formatting the y-axis as dates
    date_format = mdates.DateFormatter('%H:%M', tz=mdates.UTC)
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(date_format)
    
    ax.set_ylabel("Capture start time (UTC)")
    ax.set_xlabel("Capture interval (ms)")
    ax.set_xlim(xlim)
    ax.invert_yaxis()
    
    # Colorbar
    fig.colorbar(im, label='Normalized ADC value', fraction=0.046, pad=0.04)
    
    # Set title and save the plot
    ax.set_title("Heatmap")
    return fig, ax

def gen_chu_tick(when, sample_rate=20000, duration=0.36):
    if isinstance(when, np.datetime64):
        when = dt.datetime.utcfromtimestamp(when.tolist()/1e9)

    
    if not isinstance(when, dt.datetime):
        raise Exception("Not a datetime")


    if duration > 1:
        raise Exception("Duration must be at most 1 second")

    count = int(duration * sample_rate)
    t = np.arange(0, duration, 1/float(sample_rate))
    y_base = np.zeros(t.shape[0])
    data_duration = 0
    
    if when.second >= 31 and when.second <= 39:
        data_duration = 0.01
        end = int(np.round(sample_rate * 0.01))
        if (end > count):
            end = count
        y_sin = np.sin(2 * np.pi * 1000 * t[:end])
        y_base = y_base + zpad(y_sin, count)
    elif when.second == 0:
        data_duration = duration
        end = int(np.round(sample_rate * duration))
        if (end > count):
            end = count
        y_sin = np.sin(2 * np.pi * 1000 * t[:end])
        y_base = y_base + zpad(y_sin, count)
    elif when.minute == 0 and when.second >= 1 and when.second <= 10:
        pass
    elif when.second == 29:
        pass
    elif when.second >= 51 and when.second <= 59:
        data_duration = 0.01
        end = int(np.round(sample_rate * 0.01))
        if (end > count):
            end = count
        y_sin = np.sin(2 * np.pi * 1000 * t[:end])
        y_base = y_base + zpad(y_sin, count)
    else:
        data_duration = 0.3
        end = int(np.round(sample_rate * 0.3))
        if (end > count):
            end = count
        y_sin = np.sin(2 * np.pi * 1000 * t[:end])
        y_base = y_base + zpad(y_sin, count)

    return (t, y_base)

def correlate2d_rows(a, b, mode="full"):
    return np.array([scipy.signal.correlate(a[i, :], b[i, :], mode=mode) for i in range(a.shape[0])])

def to_netcdf(file):
    data = load_file(file)
    filename = os.path.basename(file)
    data.to_netcdf(filename + ".nc", format="NETCDF4")

def zpad(arr, target_length):
    """
    Pads a numpy array with zeros to a given target length.

    Parameters:
    arr (numpy.ndarray): Input array to pad.
    target_length (int): The target length of the output array.

    Returns:
    numpy.ndarray: A new array padded with zeros to the target length.
    """
    padding_length = max(target_length - arr.size, 0)
    return np.pad(arr, (0, padding_length), 'constant')

def hex_to_rgba(hex_color):
    """
    Convert a hex color string to an RGBA tuple.
    
    Parameters:
    - hex_color: A string representing the hex color code (e.g., "#RRGGBB", "#RRGGBBAA", "#RGB", or "#RGBA").
    
    Returns:
    - A tuple of (red, green, blue, alpha), where red, green, and blue are integers in [0, 255], 
      and alpha is a float in [0, 1].
    """
    hex_color = hex_color.lstrip('#')
    length = len(hex_color)
    
    if length == 3:
        # Expand shorthand format to standard format
        hex_color = ''.join([c*2 for c in hex_color])
        length = 6
    
    if length == 4:
        # Expand shorthand format with alpha to standard format
        hex_color = ''.join([c*2 for c in hex_color])
        length = 8
    
    if length == 6:
        # No alpha value provided, default to 255 (fully opaque)
        hex_color += 'FF'
    
    # Convert hex values to integers
    red, green, blue, alpha = [int(hex_color[i:i+2], 16) for i in range(0, 8, 2)]
    reg, green, blue = red/255, green/255, blue/255
    
    # Normalize alpha to [0, 1]
    alpha = alpha / 255.0
    
    return (red, green, blue, alpha)
