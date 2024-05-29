import sys
sys.path.append("..")
from deepface import DeepFace
from detectron2.data.detection_utils import pil_image_to_numpy


class FaceAlgo:
    
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

        return face_objs

if __name__ == "__main__":
    from PIL import Image
    m = FaceAlgo()  # pragma: no cover

    image = Image.open("./face.jpg")
    out = m.predict(image)
    print(out)