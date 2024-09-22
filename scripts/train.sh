

neuralnet_name=$1
accelerate launch --mixed_precision fp16 main.py --batch_size=8 --epochs=300 --learning_rate=0.0001 --train_or_test=train --neuralnet_name=${neuralnet_name}

