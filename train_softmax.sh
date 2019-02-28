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
IMAGE_HEIGHT=112
IMAGE_WIDTH=112
EMBEDDING_SIZE=1024
LR_FILE=/workspace/project/MassFace/lr_coco.txt
OPT=ADAM
FC_BN='--fc_bn'

DATA_DIR='/workspace/data/face/webface_112x112' # data located in hard disk

NAME=${NETWORK}_${LOSS_TYPE}_${CROP}_${GPU}_${SCALE}_${WEIGHT}_${ALPHA}_${OPT}_${FC_BN}_${IMAGE_WIDTH}_${EMBEDDING_SIZE}
SAVE_DIR=/workspace/saved/softmax

CMD="\" bash -c 'python /workspace/project/MassFace/train/train_softmax.py --logs_base_dir ${SAVE_DIR}/logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/$NAME/ --data_dir ${DATA_DIR}  --optimizer ${OPT} --learning_rate -1 --max_nrof_epochs 100 --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus ${NUM_GPUS} --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE} --scale ${SCALE} --weight ${WEIGHT} --alpha ${ALPHA} --network ${NETWORK} ${FC_BN} --image_height ${IMAGE_HEIGHT} --image_width  ${IMAGE_WIDTH} --embedding_size ${EMBEDDING_SIZE}'\""
LOCAL_CMD="python /workspace/project/MassFace/train/train_softmax.py --logs_base_dir ${SAVE_DIR}/logs/${NAME}/ --models_base_dir ${SAVE_DIR}/models/$NAME/ --data_dir ${DATA_DIR}  --optimizer ${OPT} --learning_rate -1 --max_nrof_epochs 100 --random_flip --learning_rate_schedule_file ${LR_FILE}  --num_gpus ${NUM_GPUS} --weight_decay ${WEIGHT_DECAY} --loss_type ${LOSS_TYPE} --scale ${SCALE} --weight ${WEIGHT} --alpha ${ALPHA} --network ${NETWORK} ${FC_BN} --image_height ${IMAGE_HEIGHT} --image_width  ${IMAGE_WIDTH} --embedding_size ${EMBEDDING_SIZE}"

echo ${LOCAL_CMD} && eval ${LOCAL_CMD}
