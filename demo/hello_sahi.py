import os
os.getcwd()


# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from IPython.display import Image


# Select a model to use, we use a DETR model.
model_path = "facebook/detr-resnet-50"

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')


detection_model = AutoDetectionModel.from_pretrained(
    model_type='huggingface',
    model_path=model_path,
    config_path=model_path,
    confidence_threshold=0.5,
    image_size=640,
    device="cpu", # or 'cuda'
)


result = get_prediction("/home/mandip/Github/sahi/demo/demo_data/M6J33.jpeg", detection_model)

#result = get_prediction(read_image("demo_data/M6 377 2A J33.jpeg"), detection_model)

result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")



result = get_sliced_prediction(
    "demo_data/M6J33.jpeg",
    detection_model,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)

result.export_visuals(export_dir="demo_data/")

Image("demo_data/prediction_visual.png")

object_prediction_list = result.object_prediction_list


object_prediction_list[:20]