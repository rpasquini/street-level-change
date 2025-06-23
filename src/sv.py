import pandas as pd
from streetview import search_panoramas
from tqdm import tqdm


def get_panos(points_gdf):
    panos = []
    for _, row in tqdm(points_gdf.iterrows(), total=len(points_gdf)):
        lat, lon = row["geometry"].y, row["geometry"].x
        panos_there = search_panoramas(lat=lat, lon=lon)

        for pano in panos_there:
            id = pano.pano_id
            lat, lon = pano.lat, pano.lon
            date = pano.date
            panos.append((id, lat, lon, date))

    panos = pd.DataFrame(panos).drop_duplicates().reset_index(drop=True)
    return panos
