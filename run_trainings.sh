#!/bin/bash


python main.py -c configs/flower17_efficientnet_config.json
python main.py -c configs/flower17_mobilenetV2_config.json
python main.py -c configs/flower17_shuffleNet_config.json
python main.py -c configs/flower17_squeezeNet_config.json

python main.py -c configs/flower102_efficientnet_config.json
python main.py -c configs/flower102_mobilenetV2_config.json
python main.py -c configs/flower102_shuffleNet_config.json
python main.py -c configs/flower102_squeezeNet_config.json

#python main.py -c  configs/plantCLEF2015_efficientnet_config.json
#python main.py -c  configs/plantCLEF2015_mobilenetV2_config.json
#python main.py -c  configs/plantCLEF2015_shuffleNet_config.json
#python main.py -c  configs/plantCLEF2015_squeezeNet_config.json





