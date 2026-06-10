from human_presence_detection import HumanPresenceDetection
from human_tracking import HumanTracking
from idle_attention import IdleAttention
import rospy
from kinematics.kinematic_interface import QTrobotKinematicInterface
import threading
from riva_node import run_riva_node
from qt_deep_face import QTDeepFace, run_deep_face

class QTEmotionBot:
    def __init__(self):
        #kinematics
        self.kinematics = QTrobotKinematicInterface()
        self.active_speaker = None        # was missing
        self.robot_attention_pos = None
        rospy.sleep(1.0)  

        self.deep_face = QTDeepFace(actions=['emotion'])

        #head tracking stuff
        self.human_detector = HumanPresenceDetection(detection_framerate=15, external_vad_trigger=True, kinematic_interface=self.kinematics, emotion_analyzer=self.deep_face)
        self.human_detector.register_callback(self._human_presence_callback)
        self.human_tracker = HumanTracking(human_detector=self.human_detector)
        self.idle_attention = IdleAttention(attention_time=1, human_tracker=self.human_tracker)

        
    def start(self):
        self._reset_interaction()
        #start the thread for riva speech to text
        riva_thread = threading.Thread(target=run_riva_node, args=())
        riva_thread.daemon = True
        riva_thread.start()
        # #start the thread for Deep face
        # deep_face_thread = threading.Thread(target=run_deep_face, args=())
        # deep_face_thread.daemon = True
        # deep_face_thread.start()


        rospy.spin()
        

    def _reset_interaction(self):
        rospy.loginfo("Reseting interaction to idle")
        self.robot_attention_pos = None
        self.human_tracker.untrack()
        self.idle_attention.start()

    def _human_presence_callback(self, persons):
        # if voice and faces:        
        for id, person in persons.items():
            if person.get('voice'):                
                self.active_speaker = person
    



if __name__ == "__main__":
    rospy.init_node("emotional_bot")
    rospy.loginfo("Emotional Robot starting...")
    rospy.sleep(1)
    emotional_bot = QTEmotionBot()
    emotional_bot.start()