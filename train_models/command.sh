


python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.001 --lr-decay 0.5 --name nmp --imagenet

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.2 --name nmp --imagenet

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.5 --name nmp --imagenet --augment

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.001 --lr-decay 0.5 --name nmp --imagenet --augment

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.2 --name nmp --imagenet --augment


python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.001 --lr-decay 0.5 --name nmp --imagenet

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --name nmp --imagenet

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.5 --name nmp --imagenet --augment

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.001 --lr-decay 0.5 --name nmp --imagenet --augment

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --name nmp --imagenet --augment




python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --name nmp --topn-class 7 --imagenet --augment --no-weight-class

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --name nmp --topn-class 7 --imagenet --augment --weight-class

python train_models.py --tensorboard --epochs 20 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --name nmp --topn-class 7 --augment --weight-class


python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --imagenet --augment --no-weight-class

python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --imagenet --augment --weight-class

python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --augment --weight-class



python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --imagenet --augment --no-weight-class

python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --imagenet --augment --weight-class

python train_models.py --tensorboard --epochs 60 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.01 --lr-decay 0.2 --decay-every 10 --name nmp --topn-class 7 --augment --weight-class



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --augment --no-weight-class

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --augment --weight-class-log

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --no-weight-class

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.01 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-log



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tunelast --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

#python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

#python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tunelast --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

#python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

#python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tunelast --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tuneclassifier --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tuneclassifier --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear




python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear --weight-decay 0.002

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear --weight-decay 0.01

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear --weight-decay 0.005

python train_models.py --topn-class 6 --imagenet --augment --test nmp_rodent_256_scale_topcls-6_resnet18_pretrained_tuneall_arg-True_wtclass-linear_imgnetnorm-True_drop-0_lr0.0004_decay-8-0.2_2017-12-15-12-07

#not run
python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear




python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear


python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg13bn_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --no-weight-class

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34 --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear






python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet18 --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear



python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear


python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tuneclassifier --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg16bn_pretrained_tuneclassifier --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

#maybe use 0.01 lr to tune last layer

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tunelast --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model resnet34_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear


python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tunelast --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tunelast --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tuneall --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tuneall --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear


python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tuneclassifier --lr 0.002 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear

python train_models.py --tensorboard --epochs 30 -b 64 --print-freq 10 --dataset rodent_256_scale --model vgg19bn_pretrained_tuneclassifier --lr 0.0004 --lr-decay 0.2 --decay-every 8 --name nmp --topn-class 6 --imagenet --augment --weight-class-linear
