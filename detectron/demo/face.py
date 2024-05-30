import sys
sys.path.append("..")
from deepface import DeepFace
from detectron2.data.detection_utils import pil_image_to_numpy
import numpy as np
import time

from PIL import Image

class FaceAlgo:
    
    need_save_image = False

    backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'fastmtcnn',
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
    ]

    recognition_models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        "GhostFaceNet",
    ]

    distance_metric = [
        'cosine',
        'euclidean',
        'euclidean_l2',
    ]

    def __init__(self,need_save_image=False):
        self.need_save_image = need_save_image

    def np_to_pil(self,np_img):
        # 转换 BGR 到 RGB
        # rgb_image = np_img[:, :, ::-1]
        np_img = (np_img * 255).astype(np.uint8)
        # print(rgb_image.shape)
        #convert numpy array to PIL Image
        return Image.fromarray(np_img)


    def predict(self,pil_image,pil_image1=None,algo_type="detect"):
        image = pil_image_to_numpy(pil_image)
        if pil_image1 is not None:
            image1 = pil_image_to_numpy(pil_image1)

        if algo_type == "detect":
            return self.detect(image)
        elif algo_type =="recognize":
            return self.recognition(image)
        elif algo_type =="compare":
            return self.verify(image,image1)
        elif algo_type =="feature":
            return self.embeddings(image)
        elif algo_type =="attr":
            return self.analysis(image)

    def verify(self,a,b):
        #face verification
        obj = DeepFace.verify(
            img1_path = a, 
            img2_path = b, 
            detector_backend = self.backends[0],
        )

        return obj

    def recognition(self,a):
        #face recognition
        dfs = DeepFace.find(
            img_path = a, 
            db_path = "./", 
            detector_backend = self.backends[1],
        )

        return dfs

    def embeddings(self,a):
        #embeddings
        embedding_objs = DeepFace.represent(
            img_path = a, 
            detector_backend = self.backends[2],
        )

        return embedding_objs
    
    def analysis(self,a):
        #facial analysis
        demographies = DeepFace.analyze(
            img_path = a, 
            detector_backend = self.backends[3],
        )

        return demographies

    def detect(self,a):
        #face detection and alignment
        face_objs = DeepFace.extract_faces(
            img_path = a, 
            detector_backend = self.backends[4],
        )

        ret = []
        faces = []
        # 获取当前时间戳
        current_timestamp = time.time()
        for i,obj in enumerate(face_objs):
            print(obj['face'])
            face_image = self.np_to_pil(obj['face'])
            if self.need_save_image:
                face_image.save(f"{current_timestamp}_{i}.png")
            item = {'facial_area':obj['facial_area'],'confidence':obj['confidence']}
            ret.append(item)
            faces.append(face_image)
        return ret,faces

# if __name__ == "__main__":
    
#     m = FaceAlgo()  # pragma: no cover

#     image = Image.open("./face1.jpeg")
#     out = m.predict(image)
#     print(out)