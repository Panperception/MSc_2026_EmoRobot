import threading
import time
import collections

class EmotionBuffer:
    
    def __init__(self, max_duration = 600):
        self.lock = threading.Lock()
        self.entries = collections.deque()
        self.max_duration = max_duration

    def add(self, dominant, confidence, scores):
        entry = {
            "timestamp": time.time(),
            "dominant": dominant,
            "confidence": confidence,
            "scores": scores
        }
        #print(f"Creating entry: {entry}")
        with self.lock:
            self.entries.append(entry)
            cutoff = time.time() - self.max_duration
            while self.entries and self.entries[0]["timestamp"] < cutoff:
                self.entries.popleft()
    
    def get_window(self, start_time, end_time):
        """"
        Return all the entries between start and endtime
        """
        
        with self.lock:
            for e in self.entries:
                #print(f"Timestamp: {e['timestamp']}, Start: {start_time}, End: {end_time}")
                pass
            return [ 
                e for e in self.entries
                
                if start_time <= e["timestamp"] <= end_time
            ]

shared_buffer = EmotionBuffer()
