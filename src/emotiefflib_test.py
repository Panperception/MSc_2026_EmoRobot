import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

bridge = CvBridge()
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

def image_callback(msg):
    # Convert ROS Image message to OpenCV frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    # Run emotion recognition
    emotion, scores = fer.predict_emotions(frame, logits=False)
    print(f"Emotion: {emotion}, Scores: {scores}")

rospy.init_node('emotion_recognition_test')
rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
rospy.spin()