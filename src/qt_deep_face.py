#!/usr/bin/env python3
import os
import json 
import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
from emotion_buffer import shared_buffer

def run_deep_face():          
    dp = QTDeepFace(actions=['emotion'], recognize_persons=False, persons_db_path="/home/qtrobot/persons")

    rospy.loginfo(f"OpenCV enabled gpu: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    rospy.loginfo(f"TensorFlow enabled gpu: {len(tf.config.list_physical_devices('GPU')) > 0}")

    rospy.loginfo("qt_deep_face started!")  

class QTDeepFace:    
    def __init__(self, 
                 actions=['emotion'], # actions : ['age', 'gender', 'race', 'emotion'])
                 recognize_persons=False,
                 persons_db_path="./",
                 frame_rate=5,
                 model_name='VGG-Face',
                 detector_backend='ssd'): #was retinaface
        self.actions = actions  
        self.recognize_persons = recognize_persons
        self.persons_db_path = persons_db_path  
        self.frame_rate = frame_rate
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/qt_deep_face/image/out", Image, queue_size=10)
        self.face_pub = rospy.Publisher("/qt_deep_face/faces", String, queue_size=10)
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.prev_proc_time = rospy.get_time()


    def get_current_emotion(self):
        """Returns the last detected dominant emotion, or None if no face detected."""
        return getattr(self, '_last_dominant_emotion', None)

    def get_current_scores(self):
        """Returns the full emotion score dict from the last detected face."""
        return getattr(self, '_last_emotion_scores', {})


    def analyze_face(self, roi):
        """Call this with an already-cropped face image"""
        try:
            faces = DeepFace.analyze(roi,
                actions=self.actions,
                enforce_detection=False,
                detector_backend='skip',  # face already cropped, skip detection
                silent=True)
            
            if faces:
                dominant = faces[0].get('dominant_emotion')
                emotions = faces[0].get('emotion', {})
                shared_buffer.add(dominant, 1.0, emotions)
        except:
            pass

    def image_callback(self, data):
        if rospy.get_time() - self.prev_proc_time < 1.0/self.frame_rate:
            return
        self.prev_proc_time = rospy.get_time()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(str(e))
            
        rows, cols, channels = cv_image.shape
        # print(rows, cols)
        faces = []
        fs = []

        try:
            faces = DeepFace.analyze(np.array(cv_image),
                                     actions = self.actions,
                                     enforce_detection=True,
                                     silent=True,
                                     detector_backend=self.detector_backend)
        except:
            pass
        
        if self.recognize_persons:
            try:
                fs = DeepFace.find(np.array(cv_image),
                                   db_path=self.persons_db_path,
                                   enforce_detection=True,
                                   silent=True,
                                   model_name=self.model_name,
                                   detector_backend=self.detector_backend)
            except:                
                pass
    
        face_found = len(faces) > 0
        person_found = len(fs) > 0  and all(not df.empty for df in fs)

        if not face_found and not person_found: 
            #print("No face detected", flush=True)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(str(e))
            return 

        # we have detected some faces
        for i, face in enumerate(faces):
            face_confidence = face.get('face_confidence', 0.0)
            if face_confidence == 0.0:
                continue
            
            region = face.get('region', {})
            emotions = face.get('emotion', {})
            dominant = face.get('dominant_emotion', 'unkown')  
            #print("Dominant emotion: ", dominant_emotion)          
            age = face.get('age')
            x = region.get('x', 0)
            y = region.get('y', 0)
            h = region.get('h', 0)
            w = region.get('w', 0)
            eye_l = region.get('left_eye', [])
            eye_r = region.get('right_eye', [])
                        
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,255,0), 1)            
            # if eye_l:
            #     cv2.circle(cv_image, (eye_l[0], eye_l[1]), 2, (50,0,200), 2)
            # if eye_r:
            #     cv2.circle(cv_image, (eye_r[0], eye_r[1]), 2, (50,0,200), 2)

            y_offset = y + h + 10
            x_offset = x + 40
            for key, value in emotions.items():
                cv2.putText(cv_image, f"{key}:", (x, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, lineType=cv2.LINE_AA)
                cv2.rectangle(cv_image, (x_offset, y_offset+1), (x_offset+int(value/2.0), y_offset+3), (0,255,0), cv2.FILLED)
                cv2.rectangle(cv_image, (x_offset, y_offset), (x_offset+50, y_offset+4), (255,255,255), 1)
                y_offset = y_offset + 10
            self._last_dominant_emotion = dominant
            self._last_emotion_scores = emotions
            scores_str = ', '.join(f"{k}: {v:.1f}%" for k, v in sorted(emotions.items(), key=lambda x: -x[1]))
            #print(f"Timestamp ({time.time()})| Face {i+1} | confidence: {face_confidence:.2f} | dominant: {dominant} | {scores_str}", flush=True)
            shared_buffer.add(dominant, face_confidence, emotions)
            if age:
                print(f"       age: {age}", flush=True)
                cv2.putText(cv_image, f"age: {age}", (x, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, lineType=cv2.LINE_AA)
                y_offset = y_offset + 10

        # we have recognized some persons         
        for df in fs:
            if df.empty :
                continue            
            face = df.iloc[0]
            identity = face.get('identity')            
            base_name = os.path.basename(identity)
            name = base_name.split('_')[0]
            x = face.get('source_x', 0)
            y = face.get('source_y', 0)
            h = face.get('source_h', 0)
            w = face.get('source_w', 0)
            distance = face.get('distance')
            # print(name, distance, x, y, h, w)            
            # cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0,255,0), 1)
            y_offset = y + h + -10
            x_offset = x + 5
            cv2.putText(cv_image, f"{name}", (x_offset, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, lineType=cv2.LINE_AA)
            faces.append( {
                'name': name,
                'identity': identity,
                'distance': float(distance),
                'threshold': float(face.get('threshold')),
                "region": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                    }
                })            
        
        self.face_pub.publish(json.dumps(faces))

        # # publish image
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(str(e))



