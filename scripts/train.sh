

neuralnet_name=$1
accelerate launch --mixed_precision fp16 main_v.py --batch_size=8 --epochs=300 --learning_rate=0.0001 --train_or_test=train --neuralnet_name=${neuralnet_name}


# run locally
accelerate launch --mixed_precision fp16 main_v.py --batch_size=8 --epochs=50 --learning_rate=0.0001 --train_or_test=train --neuralnet_name=unet_small

accelerate launch --mixed_precision fp16 main_v.py --batch_size=8 --epochs=50 --learning_rate=0.0001 --train_or_test=test --neuralnet_name=unet_small --load_checkpoint


accelerate launch --mixed_precision fp16 main_t2i.py --dataset Multi-Modal-CelebA-HQ --batch_size=8 --epochs=3 --learning_rate=0.0001 --train_or_test=train --neuralnet_name=unet_small
