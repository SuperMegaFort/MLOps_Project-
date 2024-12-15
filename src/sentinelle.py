import os
import getpass
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import storage
import csv
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
)

# Configuration SentinelHub avec credential.csv
config = SHConfig()

with open('/Users/cyriltelley/Desktop/MSE/TSM_MachLeData/MLOps_Project/credential_sentinelle.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        config.sh_client_id = row['sentinelHub_client_id']
        config.sh_client_secret = row['sentinelHub_client_secret']

config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save("cdse")

# Configuration Google Cloud Storage
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "chemin/vers/votre/fichier/credentials.json" 
storage_client = storage.Client()
bucket_name = "input-image"  # Nom de votre bucket
bucket = storage_client.bucket(bucket_name)

# Coordonnées pour différents types de terrains
regions = {
    "urban": (6.616746, 46.512455, 6.670934, 46.551785),       
    "forest": (6.979621, 46.950768, 7.048630, 46.990702),      
    "agriculture": (6.558203, 46.724210, 6.682192, 46.762345), 
    "water": (6.448184, 46.428753, 6.526922, 46.456451),      
    "mountain": (7.974087, 46.095684, 8.063164, 46.146728),    
}

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [ { bands: ["B02", "B03", "B04"] } ],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""
output_dir = r"/Users/cyriltelley/Desktop/MSE/TSM_MachLeData/MLOps_Project/salelite_image"
os.makedirs(output_dir, exist_ok=True)

existing_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
total_images = len(existing_files)  

brightness_factor = 3.5 / 255  

for region_name, coords in regions.items():
    print(f"Téléchargement pour la région : {region_name}...")
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    resolution = 60  # Résolution
    size = bbox_to_dimensions(bbox, resolution=resolution)

    print(f"Boîte englobante : {coords}, Taille en pixels : {size}")
    request = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=("2020-07-13", "2024-07-14"),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )


    try:
        true_color_imgs = request.get_data()
        print(f"Téléchargé {len(true_color_imgs)} images pour la région {region_name}.")

        
        for idx, img in enumerate(true_color_imgs):
            total_images += 1  
            img_corrected = img * brightness_factor
            img_corrected = img_corrected.clip(0, 1)
            img_corrected = (img_corrected * 255).astype('uint8')
            img_resized = Image.fromarray(img_corrected)
            img_resized = img_resized.resize((64, 64), Image.Resampling.LANCZOS)
            output_path = os.path.join(output_dir, f"{region_name}_{total_images}.jpg")
            img_resized.save(output_path, "JPEG", dpi=(96, 96))
            print(f"Image sauvegardée : {output_path}")

            blob_name = f"{region_name}_{total_images}.jpg"  # Nom du fichier dans le bucket
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(output_path)  # Upload depuis le chemin local
            print(f"Image envoyée vers gs://{bucket_name}/{blob_name}")


            

    except Exception as e:
        print(f"Erreur lors du téléchargement pour {region_name}: {e}")


print(f"Téléchargement terminé. {total_images} images ont été sauvegardées dans :", output_dir)
