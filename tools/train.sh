export DETECTRON2_DATASETS=/projects/iiitd/mrcnn/data/
python -W ignore train_net.py --num-gpus 8 --config-file "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
