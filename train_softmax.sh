#NETWORK=sphere_network
#NETWORK=resface
#NETWORK=inception_net
#NETWORK=resnet_v2
NETWORK=mobilenet

CROP=112
echo $NAME
GPU=2
#GPU=0,1,2,3
NUM_GPUS=1
ARGS="CUDA_VISIBLE_DEVICES=${GPU}"
#WEIGHT_DECAY=1e-3
WEIGHT_DECAY=1e-4
#LOSS_TYPE=cosface
LOSS_TYPE=softmax
SCALE=64.
#WEIGHT=3.
#SCALE=32.
WEIGHT=2.
#WEIGHT=2.5
#ALPHA=0.35
ALPHA=0.25
#ALPHA=0.2
#ALPHA=0.3
#LR_FILE=lr_coco.txt
IMAGE_HEIGHT=112
IMAGE_WIDTH=112
EMBEDDING_SIZE=1024
LR_FILE=/workspace/project/MassFace/lr_coco.txt
OPT=ADAM
#OPT=MOM
FC_BN='--fc_bn'




#DATA_DIR='local_dataset/webface_112x112'
DATA_DIR='/workspace/data/face/webface_112x112' # data located in hard disk
#DATA_DIR='dataset/debug'
#PRETRAINED_MODEL="pretrained_model/model-20181222-102734.ckpt-60000"
#PRETRAINED_MODEL='mobilenet_model/model-20181226-131708.ckpt-43200'

NAME=${NETWORK}_${LOSS_TYPE}_${CROP}_${GPU}_${SCALE}_${WEIGHT}_${ALPHA}_${OPT}_${FC_BN}_${IMAGE_WIDTH}_${EMBEDDING_SIZE}
SAVE_DIR=/workspace/saved/softmax

CMD="\" bash -c 'python /workspace/project/MassFace/train/train_softmax.py --logs_base_dir ${SAVE_DIR}/logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/$NAME/ --data_dir ${DATA_DIR}  --optimizer ${OPT} --learning_rate -1 --max_nrof_epochs 100 --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus ${NUM_GPUS} --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE} --scale ${SCALE} --weight ${WEIGHT} --alpha ${ALPHA} --network ${NETWORK} ${FC_BN} --image_height ${IMAGE_HEIGHT} --image_width  ${IMAGE_WIDTH} --embedding_size ${EMBEDDING_SIZE}'\""
LOCAL_CMD="python /workspace/project/MassFace/train/train_softmax.py --logs_base_dir ${SAVE_DIR}/logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/$NAME/ --data_dir ${DATA_DIR}  --optimizer ${OPT} --learning_rate -1 --max_nrof_epochs 100 --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus ${NUM_GPUS} --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE} --scale ${SCALE} --weight ${WEIGHT} --alpha ${ALPHA} --network ${NETWORK} ${FC_BN} --image_height ${IMAGE_HEIGHT} --image_width  ${IMAGE_WIDTH} --embedding_size ${EMBEDDING_SIZE}"

#LOCAL_CMD="python /workspace/project/TripletFace/train.py --logs_base_dir ${SAVE_DIR}logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/${NAME}/  --image_size 224  --optimizer ADAGRAD --learning_rate 0.001 --weight_decay 1e-4 --max_nrof_epochs 10000  --network ${NETWORK} --dataset ${DATASET} --data_dir ${DATA_DIR} --pretrained_model ${PRETRAINED_MODEL} --random_crop --random_flip --image_size 112 --strategy ${STRATEGY} --mine_method ${MINE_METHOD} --num_gpus 1 --embedding_size 1024 --scale 10 --people_per_batch ${P} --images_per_person ${K}"
echo ${LOCAL_CMD} && eval ${LOCAL_CMD}
cmd="axer create --name='test_lip_cos' --cmd=${CMD} --gpu_count='1' --image='CV-Caffe_TF1.8-Py3' --prior_gpu_kind='V100' --project_id 332"
#echo ${cmd}  
#echo ${cmd}  && eval ${cmd}
