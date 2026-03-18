import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def compute_distance(data, lon_column, lat_column, invert=False):
    start_lon = data[lon_column].iloc[0]
    start_lat = data[lat_column].iloc[0]

    distance = data.apply(
        lambda row: haversine(
            start_lon, start_lat,
            row[lon_column], row[lat_column]
        ) * 100,
        axis=1
    )

    if invert:
        distance = -distance
        distance = distance - distance.min()  # shift so min == 0

    return distance

def plot_scatter(csv_file, x_column, y_column, y1_color='b', y2_color='r', invert_distance=False):
    data = pd.read_csv(csv_file)

    lon_column = "Longitude"
    lat_column = "Latitude"

    required = {x_column, y_column, lon_column, lat_column}
    if not required.issubset(data.columns):
        print(f"Missing required columns: {required - set(data.columns)}")
        return

    data['distance'] = compute_distance(
        data, lon_column, lat_column, invert=invert_distance
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.scatter(data[x_column], data[y_column],
                color=y1_color, s=0.8, label=y_column)
    ax2.scatter(data[x_column], data['distance'],
                color=y2_color, s=0.8, label='distance')

    ax1.set_xlabel(x_column)
    ax1.set_ylabel(y_column, color=y1_color)
    ax2.set_ylabel("distance", color=y2_color)

    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)

    fig.suptitle(f"{y_column} (left) and distance (right) vs {x_column}")

    fig.tight_layout()
    plt.show()

def plot_line(csv_file, x_column, y_column, y1_color='b', y2_color='r', invert_distance=False):
    data = pd.read_csv(csv_file)

    lon_column = "Longitude"
    lat_column = "Latitude"

    required = {x_column, y_column, lon_column, lat_column}
    if not required.issubset(data.columns):
        print(f"Missing required columns: {required - set(data.columns)}")
        return

    data['distance'] = compute_distance(
        data, lon_column, lat_column, invert=invert_distance
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(data[x_column], data[y_column],
             color=y1_color, label=y_column)
    ax2.plot(data[x_column], data['distance'],
             color=y2_color, label='distance')

    ax1.set_xlabel(x_column)
    ax1.set_ylabel(y_column, color=y1_color)
    ax2.set_ylabel("distance", color=y2_color)

    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)

    fig.suptitle(f"{y_column} (left) and distance (right) vs {x_column}")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot CSV data with Matplotlib.')
    parser.add_argument('logfile', type=str, help='Log file for CSV')
    parser.add_argument('-x', '--x_axis', default="time", help='Graph x axis')
    parser.add_argument('-y', '--y_axis', required=True, help='Graph y axis')
    parser.add_argument('-t', '--graph_type', default="scatter",
                        help='scatter or line')
    parser.add_argument('--y1_color', default='b', help='y1 axis color')
    parser.add_argument('--y2_color', default='r', help='y2 axis color')

    parser.add_argument('--invert-distance', action='store_true',
                        help='Invert the distance plot')

    args = parser.parse_args()

    if args.graph_type == "scatter":
        plot_scatter(
            args.logfile, args.x_axis, args.y_axis,
            y1_color=args.y1_color,
            y2_color=args.y2_color,
            invert_distance=args.invert_distance
        )
    elif args.graph_type == "line":
        plot_line(
            args.logfile, args.x_axis, args.y_axis,
            y1_color=args.y1_color,
            y2_color=args.y2_color,
            invert_distance=args.invert_distance
        )
    else:
        raise ValueError("Invalid graph type (use scatter or line)")

