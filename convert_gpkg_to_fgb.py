import geopandas as gpd
import fiona

# for layer in fiona.listlayers("taxi_data.gpkg"):
gdf = gpd.read_file("taxi_data.gpkg", layer="pickup_points")
gdf.to_file("pickup_points_smol.fgb", driver="FlatGeobuf")
