import os
from typing_extensions import Literal
from typing import List, Any, Dict
import cv2
import json
from dotenv import load_dotenv
import torch
import supervisely as sly

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from src.demo_data import prepare_weights

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
prepare_weights()  # prepare demo data automatically for convenient debug

# code for detectron2 inference copied from official COLAB tutorial (inference section):
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html


class MyModel(sly.nn.inference.InstanceSegmentation):
    def load_on_device(
        self,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        with open(os.path.join(self.location, "model_info.json"), "r") as myfile:
            architecture = json.loads(myfile.read())["architecture"]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(architecture))
        cfg.MODEL.DEVICE = device  # learn more in torch.device
        cfg.MODEL.WEIGHTS = os.path.join(self.location, "weights.pkl")
        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes")
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionMask]:
        confidence_threshold = settings.get("confidence_threshold", 0.5)
        image = cv2.imread(image_path)  # BGR

        ####### CUSTOM CODE FOR MY MODEL STARTS (e.g. DETECTRON2) #######
        outputs = self.predictor(image)  # get predictions from Detectron2 model
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()
        ####### CUSTOM CODE FOR MY MODEL ENDS (e.g. DETECTRON2)  ########

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            # filter predictions by confidence
            if score >= confidence_threshold:
                results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results


model_dir = sly.env.folder()
print("Model directory:", model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

settings = {"confidence_threshold": 0.7}
m = MyModel(location=model_dir, custom_inference_settings=settings)
m.load_on_device(device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    # for local development and debugging
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path)
    print(f"predictions and visualization have been saved: {vis_path}")
