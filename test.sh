MODEL_DIR=models/model-20190214-150620.ckpt-600000
TEST_DATA=dataset/lfw_112x112
EMBEDDING_SIZE=1024  
FC_BN='--fc_bn'
PREWHITEN=''
IMAGE_WIDTH=112
#NETWORK=resnet50
NETWORK=mobilenet
IMAGE_HEIGHT=112
#if you have gpu
#CUDA_VISIBLE_DEVICES=1 python test/test.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type ${NETWORK} --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH}
#if you don't have gpu
CUDA_VISIBLE_DEVICES="" python test/test.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type ${NETWORK} --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH}
