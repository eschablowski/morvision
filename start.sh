cat coco_new_template.config | sed 's/PATH_CONFIG/`pwd`/' > coco_new.config
cat coco_template.config | sed 's/PATH_CONFIG/`pwd`/' > coco.config
cd models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 object_detection/model_main.py --pipeline_config_path=../../coco_new.config --model_dir=../../ve-model --num_train_steps=10000 --sample_1_of_n_eval_examples=1  --alsologtostderr &
tensorboard --logdir ../../ve-model
