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

# code for detectron2 inference copied from official COLAB tutorial (inference section):
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5


class MyModel(sly.nn.inference.InstanceSegmentation):
    def __init__(self, model_dir: str = None):
        super().__init__(model_dir)

        # Initialize Detectron2 model from config
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_weights.pkl")

        self.predictor = DefaultPredictor(cfg)
        self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get(
            "thing_classes"
        )
        print("model has been successfully deployed")

    def get_classes(self) -> list[str]:
        return self.class_names  # ["cat", "dog", ...]

    def predict(
        self, image_path: str, confidence_threshold: float = 0.8
    ) -> list[sly.nn.PredictionMask]:
        image = cv2.imread(image_path)  # BGR

        # get predictions from Detectron2 model
        outputs = self.predictor(image)
        pred_classes = outputs["instances"].pred_classes.detach().numpy()
        pred_class_names = [self.class_names[pred_class] for pred_class in pred_classes]
        pred_scores = outputs["instances"].scores.detach().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.detach().numpy()

        results = []
        for score, class_name, mask in zip(pred_scores, pred_class_names, pred_masks):
            if score < confidence_threshold:
                continue
            results.append(sly.nn.PredictionMask(class_name, mask, score))
        return results


if sly.is_production():
    # code below is running on Supervisely platform in production
    m = MyModel()
    m.serve()
else:
    api = sly.Api()
    team_id = int(os.environ["context.teamId"])
    sly.app.development.connect_to_supervisely_vpn_network()
    sly.app.development.create_debug_task(team_id, port="8000")
    exit(0)

    # user_name = users.me
    # session_name = <user_name>+development
    # ecosystem.list filter=(slug=="supervisely-ecosystem/while-true-script-v2") -> module_id
    # check app task exists
    # apps.list (team_id, filter=[module_id], onlyRunning=True) ->  [apps info with field tasks info list for every app]
    # check session name in tasks info.meta.name

    # if app task exists - false
    # api.task.start(module_id=777) (one of app_id or module_id)

    team_id = int(os.environ["context.teamId"])

    # api.app.get_info_by_id(id=current_app.id).config.get('modalTemplateState', {})

    api = sly.Api()
    # wg-quick must be run as root. Please enter the password for max to continue:
    # TODO: create HeadlessApplication class
    # TODO: get app module id by name? - rename in UI
    # TODO: get app id by name
    # TODO: run app on agent that do not runinng
    # ui app id -> app module
    # how to get app id that do not installed in team?
    # how to run on fake agent?
    # api.task.start - add redirect_requests argument
    # tasks.run.app - can i use moduleId? get module info by name?
    # tasks.run.app - team_id? instead of workspace_id???
    # redirect_requests = {"token": "<API TOKEN>", "port": 8000}

    # sly-net.sh up
    # run while-true app with redirect_requests -> task_id (check existance) - try fake task run on fake hidden agent??
    # debugging
    # sly-net.sh down
    # kill while-true app? - txt file in SDK? .debug-app.pid (hidden)

    # task_info = api.task.get_info_by_id(19789)
    # app_info = api.app.get_info_by_id(6609)

    task_info = api.task.start(
        agent_id=17, app_id=6609, workspace_id=619, task_name="test vpn"
    )
    api.task.wait(19791, target_status=api.task.Status.FINISHED)
    x = 10

    # for local development and debugging
    model_dir = os.path.join(os.getcwd(), "my_model")
    m = MyModel(model_dir)

    # debug local image
    image_path = os.path.join(os.getcwd(), "demo_data/image_01.jpg")
    results = m.predict(image_path)
    vis_path = os.path.join(os.getcwd(), "demo_data/image_01_prediction.jpg")
    m.visualize(results, image_path, vis_path)
    print("predictions and visualization have been created")
