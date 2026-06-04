import pandas as pd
import geopandas as gpd

taxi_data = pd.read_csv("./yellow_tripdata_2015-01.csv")

# Vectorized bbox filter — no Python loop
mask = (
    (taxi_data["pickup_longitude"] > -74.07)
    & (taxi_data["pickup_longitude"] < -73.83)
    & (taxi_data["dropoff_longitude"] > -74.07)
    & (taxi_data["dropoff_longitude"] < -73.80)
    & (taxi_data["pickup_latitude"] > 40.70)
    & (taxi_data["pickup_latitude"] < 40.85)
    & (taxi_data["dropoff_latitude"] > 40.70)
    & (taxi_data["dropoff_latitude"] < 40.85)
)
filtered = taxi_data[mask].copy()

# Vectorized Point construction
filtered["start_point"] = gpd.points_from_xy(
    filtered["pickup_longitude"], filtered["pickup_latitude"]
)
filtered["end_point"] = gpd.points_from_xy(
    filtered["dropoff_longitude"], filtered["dropoff_latitude"]
)

gdf = gpd.GeoDataFrame(filtered, geometry="end_point", crs="EPSG:4326")

gdf.drop(columns=["start_point"]).set_geometry("end_point").set_crs(
    "EPSG:4326"
).to_file("./taxi_data.gpkg", layer="dropoff_points", driver="GPKG")
gdf.drop(columns=["end_point"]).set_geometry("start_point").set_crs(
    "EPSG:4326"
).to_file("./taxi_data.gpkg", layer="pickup_points", driver="GPKG")
