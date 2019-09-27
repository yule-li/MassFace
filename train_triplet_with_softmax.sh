cd /workspace/project/MassFace

NETWORK='mobilenet'
DATASET='webface'
#STRATEGY='min_and_min'
STRATEGY='min_and_max'
#STRATEGY='hardest'
#STRATEGY='batch_random'
#STRATEGY='batch_all'
#MINE_METHOD='simi_online'
MINE_METHOD='online'
DATA_DIR='dataset/casia-112x112'
#DATA_DIR='dataset/ASIAN_CELEB_align2_128_10000_normal:dataset/gallery_data:dataset/0_TRAIN_align2_128'
#DATA_DIR='dataset/gallery_data:dataset/0_TRAIN_align2_128'
#DATA_DIR='dataset/gallery_data_3:dataset/0_TRAIN_align2_128'
#DATA_DIR='dataset/gallery_data_3'
PRETRAINED_MODEL="pretrained_model/model-20181222-102734.ckpt-60000"
#PRETRAINED_MODEL=/workspace/saved/softmax_tal_20190904/models/mobilenet_softmax_112_2_64._2._0.25_ADAM_--fc_bn_112_1024/20190904-115526/model-20190904-115526.ckpt-60000
#P=180
#K=1
#P=18
#K=10
P=41
K=5
#P=14
#K=15
#P=10
#K=21
#P=30
#K=7
FC_BN='--fc_bn'
WITH_SOFTMAX='--with_softmax --pretrain_softmax --softmax_epoch 10 --softmax_loss_weight 0.5'
NAME=${NETWORK}_${DATASET}_${STRATEGY}_${MINE_METHOD}_${P}_${K}
SAVE_DIR=/workspace/saved/triplet_tal_20190917_with_softmax_more_3
mkdir ${SAVE_DIR}
LOCAL_CMD="python /workspace/project/MassFace/train/train_triplet.py --logs_base_dir ${SAVE_DIR}logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/${NAME}/   --optimizer ADAGRAD --learning_rate 0.001 --weight_decay 1e-4 --max_nrof_epochs 10000  --network ${NETWORK} --dataset ${DATASET} --data_dir ${DATA_DIR} --pretrained_model ${PRETRAINED_MODEL} --random_crop --random_flip --image_size 112 --strategy ${STRATEGY} --mine_method ${MINE_METHOD} --num_gpus 1 --embedding_size 1024 --scale 10 --people_per_batch ${P} --images_per_person ${K} ${FC_BN} ${WITH_SOFTMAX} 2>&1 | tee ${SAVE_DIR}/train.log7"
echo ${LOCAL_CMD} && eval ${LOCAL_CMD}
