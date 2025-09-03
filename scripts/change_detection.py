import pandas as pd
pd.set_option('display.max_columns', 999)
from src.api.streetview_fetcher import StreetViewFetcher
from src.ml.segment_street_view import ImageSegmenter
import os


CLUSTER_IDS = [254]

STEGO_MODELS_FOLDER = "./src/ml/stego/saved_models"
STEGO_MODEL_NAME = "cocostuff27_vit_base_5.ckpt"  # Path to STEGO checkpoint

model = os.path.join(STEGO_MODELS_FOLDER, STEGO_MODEL_NAME)

panos = pd.read_csv('./data/tresdefebrero/heading_fov.csv', index_col=0)
panos = panos[panos.cluster_id.isin(CLUSTER_IDS)].reset_index()

fetcher = StreetViewFetcher(data_dir=f"data/tresdefebrero/image_testing")

os.makedirs(fetcher.data_dir, exist_ok=True)

output = []
for _, row in panos.iterrows():
    panoid = row['pano_id']
    heading = row['heading']
    fov = row['fov']
    cluster_id = row['cluster_id']
    view_id = row['view_id']
    direction = row['direction']
    date = fetcher.get_panorama_metadata(panoid)["date"]
    # Fetch and save specific view using the parameters from the URL
    image_path = os.path.join(
        fetcher.data_dir, f"{cluster_id}/{view_id}.jpg"
    )
    if os.path.exists(image_path):
        print("Image exists...")
    else:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img = fetcher.get_panorama_by_id(
            panoid=panoid,
            heading=heading,
            fov=fov
        )
        img.save(image_path)

    segmenter = ImageSegmenter(model)
    metrics = segmenter.segment_image(image_path)
    class_dist = metrics["class_distribution"]
    df = pd.DataFrame({cls: vals["percentage"] for cls, vals in class_dist.items()}, index=[0])
    df['view_id'] = view_id
    df['date'] = date
    df['direction'] = direction
    df['cluster_id'] = cluster_id
    output.append(df)

output = pd.concat(output).reset_index(drop=True)
output.to_csv(f'./data/tresdefebrero/segmentation_results/{CLUSTER_IDS[0]}.csv', index=False)


"""
In this case, the building proportion increases in time when the view is looking to the south
Check the next link and see different dates:
https://www.google.com/maps/@-34.619967,-58.5562592,3a,75y,161.07h,89.34t/data=!3m8!1e1!3m6!1s-W-mrF76SN_hm--NOq3mOg!2e0!5s20240501T000000!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fcb_client%3Dmaps_sv.tactile%26w%3D900%26h%3D600%26pitch%3D0.655202285504032%26panoid%3D-W-mrF76SN_hm--NOq3mOg%26yaw%3D161.07359684441997!7i16384!8i8192?entry=ttu&g_ep=EgoyMDI1MDgzMC4wIKXMDSoASAFQAw%3D%3D
res[res.view_id.str.endswith('_S')][['date','building']]
       date   building
2   2024-05  42.813795
6   2014-07  33.433315
10  2019-06  41.163106
14  2013-12  31.890246
18  2022-06  40.777962
res[res.view_id.str.endswith('_S')][['date','building']].sort_values(by='date')
       date   building
14  2013-12  31.890246
6   2014-07  33.433315
10  2019-06  41.163106
18  2022-06  40.777962
2   2024-05  42.813795
"""