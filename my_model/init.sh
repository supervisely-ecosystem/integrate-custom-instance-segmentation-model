wget -c https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl


#@TODO: try new baselines
# wget -c https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ/42045954/model_final_ef3a80.pkl
# model_zoo.get_config_file("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py")

# https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py

# cfg = model_zoo.get_config(
#         "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
#     )
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     cfg.MODEL.DEVICE = "cpu"
#     cfg.MODEL.WEIGHTS = "my-model/model_final_ef3a80.pkl"