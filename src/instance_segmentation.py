import os
import cv2
from dotenv import load_dotenv

import supervisely as sly

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


class MyModel(sly.nn.Inference.Detection):
    def __init__(self, model_dir: str = None):
        super().__init__(model_dir)

        # Initialize Detectron2 model from config
        # learn more in detectron2 example (inference section) https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"  # autodevice
        cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_weights.pkl")

        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get(
            "thing_classes"
        )

    # def get_info():
    #     return {"a": "b", "c": "123"}

    # @property
    # def model_meta(self):
    #     if self.meta is None:
    #         classes = []
    #         for name in self.class_names:
    #             classes.append(sly.ObjClass(name, sly.Bitmap))
    #         confidence_tag = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
    #         self.meta = sly.ProjectMeta(obj_classes=classes, tag_metas=[confidence_tag])
    #     return self.meta

    def get_classes() -> List[str]:
        return self.class_names

    def predict(self, image_path) -> List[sly.nn.Inference.PredictionMask]:
        image = cv2.imread(image_path)  # BGR
        img_size = image.shape[:2]

        # Get predictions from Detectron2 model
        outputs = self.predictor(image)
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()

        # sly.nn.BBoxPrediction
        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            results.append(sly.nn.PredictionMask(class_name, mask, score))

        return results

        # # create Supervisely Labels from predicted masks
        # labels = []
        # for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
        #     if not mask.any():  # skip empty masks
        #         continue
        #     geometry = sly.Bitmap(mask)

        #     obj_class = self.model_meta.get_obj_class(class_name)
        #     conf_tag = sly.Tag(
        #         meta=self.model_meta.get_tag_meta("confidence"),
        #         value=round(float(score), 4),
        #     )
        #     label = sly.Label(geometry, obj_class, [conf_tag])
        #     labels.append(label)

        # # create Supervisely Annotation
        # annotation = sly.Annotation(img_size=img_size, labels=labels)
        # return annotation


# @TODO: path in env variable
# @TODO: ignore confidence by default
# @TODO: custom configs?
# @TODO: lazy config / new baselines?
# @TODO: specify for task specific models Detection
# @TODO: do not compile if ubuntu
# @TODO: serving modal window??? or UI

if sly.is_production():
    # code below is running on Supervisely platform in production
    m = MyModel()
    m.serve()
else:
    # for local development and debugging
    # "/usr/max/bla"  # os.path.join(os.getcwd(), "my_model")
    model_dir = os.getenv("MODEL_DIR")
    m = MyModel(model_dir)
    results = m.predict("/a/b.png")
    m.visualize(results, "/a/res.png")
