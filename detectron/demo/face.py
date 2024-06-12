import sys
sys.path.append("..")
from deepface import DeepFace
from deepface.modules import modeling
from deepface.detectors import DetectorWrapper
from detectron2.data.detection_utils import pil_image_to_numpy,convert_PIL_to_numpy
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np
import time

from PIL import Image

# print(modeling.model_obj)
# print(DetectorWrapper.face_detector_obj)

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

    def draw_face_box(self,image,face_areas):
        if not isinstance(image,np.ndarray):
            image = convert_PIL_to_numpy(image,format=None)

        image = image[:, :, ::-1]
        visualizer = Visualizer(image,instance_mode=ColorMode)
        for area in face_areas:
            x = area['x']
            y = area['y']
            w = area['w']
            h = area['h']
            visualizer.draw_box((x,y,x+w,y+h),edge_color="r")
        
        visualized_image = visualizer.get_output().get_image()
        # [:, :, ::-1]
        return Image.fromarray(visualized_image)
        
        

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
            distance_metric = self.distance_metric[0],
        )

        face_area1 = [obj['facial_areas']['img1']]
        face1 = self.draw_face_box(a,face_area1)
        face_area2 = [obj['facial_areas']['img2']]
        face2 = self.draw_face_box(b,face_area2)
        current_timestamp = time.time()
        if self.need_save_image:
                face1.save(f"{current_timestamp}_0.png")
                face2.save(f"{current_timestamp}_1.png")
        return obj,[face1,face2]

    def recognition(self,a):
        #face recognition
        dfs = DeepFace.find(
            img_path = a, 
            db_path = "./test/", 
            detector_backend = self.backends[1],
            distance_metric = self.distance_metric[0],
        )

        json_list = [df.to_json(orient='records') for df in dfs]
        top1_path = dfs[0].at[0, 'identity']
        top1_pil = Image.open(top1_path)
        return json_list,[top1_pil]

    def embeddings(self,a):
        #embeddings
        embedding_objs = DeepFace.represent(
            img_path = a, 
            detector_backend = self.backends[5],
        )
        face_areas = []
        for obj in embedding_objs:
            face_areas.append(obj['facial_area']) 
        face = self.draw_face_box(a,face_areas)
        current_timestamp = time.time()
        if self.need_save_image:
                face.save(f"{current_timestamp}.png")
        return embedding_objs,[face]
    
    def analysis(self,a):
        #facial analysis
        demographies = DeepFace.analyze(
            img_path = a, 
            detector_backend = self.backends[5],
        )

        face_areas = []
        for obj in demographies:
            face_areas.append(obj['region']) 
        face = self.draw_face_box(a,face_areas)

        current_timestamp = time.time()
        if self.need_save_image:
                face.save(f"{current_timestamp}.png")
        return demographies,[face]

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
    
#     m = FaceAlgo(need_save_image=True)  # pragma: no cover

#     image = Image.open("./test/face1.jpeg")
#     image1 = Image.open("./test/face2.jpeg")
#     out = m.predict(image,image1,algo_type="recognize")
#     print("-----------------",out)