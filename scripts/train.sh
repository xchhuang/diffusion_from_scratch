
cd examples

accelerate launch --mixed_precision fp16 main.py --batch_size=4 --epochs=100 --learning_rate=0.0001 --train_or_test=train --load_checkpoint

accelerate launch --mixed_precision fp16 main.py --batch_size=1 --epochs=2 --learning_rate=0.0001 --train_or_test=test --load_checkpoint


