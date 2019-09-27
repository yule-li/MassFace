import argparse
import os

from face_embedding import facenet_encoder

def demo(ckpt_file,img_dir):
    img_paths = [os.path.join(img_dir,i) for i in os.listdir(img_dir)]
    face_model = facenet_encoder(ckpt_file)
    feats = face_model.generate_batch_embeddings(img_paths)
    print('scores',feats.dot(feats.T))
def main():
    parser = argparse.ArgumentParser(description='demo for face recognition useage')
    parser.add_argument('--ckpt_file',default='',type=str)
    parser.add_argument('--img_dir',default='',type=str)
    args = parser.parse_args()
    demo(args.ckpt_file,args.img_dir)
     
if __name__ == '__main__':
    main()
