#!/usr/bin/env python3
import rospy
from riva_speech_recognition_vad import RivaSpeechRecognitionSilero
import time
from emotion_buffer import shared_buffer
from llm import EmotionalBot
from command_interface import CommandInterface
import re
import threading

speech_start_time = None
speech_end_time = None

TONE_MAP = {
    "gentle":       "\\vct=97\\\\rspd=82\\\\vol=65\\",   # Lower vol + slower speed = gentle (Crumpton calm)
    "calm":         "\\vct=98\\\\rspd=90\\\\vol=70\\",   # Near neutral, slightly slower
    "normal":       "\\vct=100\\\\rspd=100\\\\vol=80\\", # Acapela baseline
    "upbeat":       "\\vct=103\\\\rspd=108\\\\vol=85\\", # Slightly brighter + faster
    "enthusiastic": "\\vct=107\\\\rspd=118\\\\vol=90\\", # Higher pitch + faster + louder
}

#initlaise LLM bot
bot = EmotionalBot()

def run_riva_node():

    ci = CommandInterface()
    COMMAND_RE = re.compile(r'\{(tone|gesture|emotion):([^}]+)\}')

    def event_callback(event):
        global speech_start_time, speech_end_time
        if event == RivaSpeechRecognitionSilero.Event.RECOGNIZING:
            print("User is speaking...")
            speech_start_time = time.time()
        elif event == RivaSpeechRecognitionSilero.Event.RECOGNIZED:
            speech_end_time = time.time()
            print("User stopped speaking.")
        

    def transcript_callback(text, lang):
        print(f"User said: {text}")
        speech_end_time = time.time()
        print("Start_Time: ", speech_start_time)
        print("End_Time: ", speech_end_time)
        window = shared_buffer.get_window(start_time=speech_start_time-2.0, end_time=speech_end_time)

        window_normalised = []

        #normalise the data
        for entry in window:
            normalised_entry = {}
            for emotion in entry:
                e = entry.get(emotion)
                if emotion == "timestamp":
                    normalised_entry["timestamp"] = e
                if emotion == "dominant":
                    normalised_entry["dominant"] = e
                if emotion == "confidence":
                    normalised_entry["confidence"] = e
                if emotion == 'scores':
                    total = round(sum(e.values()), 2)
                    normalised_entry['scores'] = {score: round(value / total, 4) for score, value in e.items()}
                window_normalised.append(normalised_entry)
        
        #format the window for classification
        formatted_window = format_window_for_classification(window_normalised)
        
        #now need to pass to the LLM for processing
        response = bot.chat(text, formatted_window)
        
        print("Formatted window: ", formatted_window)
        print("LLM response: ", response)
        print("Response from response: ", response.robot_response)
        execute_response(response.robot_response, ci)
       

    def format_window_for_classification(window):
        #for each entry in the window we need to count dominant emotions
        #get average for each emotion
        #and get the peak score
        emotion_counts = {}
        emotion_averages = {}
        emotion_peaks = {}
        for entry in window:
            dominant = entry.get("dominant")
            if dominant:
                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
            
            scores = entry.get("scores", {})
            for emotion, score in scores.items():
                emotion_averages[emotion] = emotion_averages.get(emotion, 0) + score
                if emotion not in emotion_peaks or score > emotion_peaks[emotion]:
                    emotion_peaks[emotion] = score
        
        return {
            "counts": emotion_counts,
            "averages": {emotion: score / len(window) for emotion, score in emotion_averages.items()},
            "peaks": emotion_peaks
        }
    # def speak_with_tone(response_text):
    #     resolved = response_text
    #     for key, tags in TONE_MAP.items():
    #         resolved = resolved.replace(f"{{{key}}}", tags)
    #     ci._cmd_talk({'message': resolved})

    def parse_response(text):
        actions = []
        last_end = 0
        current_tone = None

        # Combine both patterns into one scan, sorted by position
        COMBINED_RE = re.compile(
            r'\{tone:(gentle|calm|normal|upbeat|enthusiastic)\}|\{(gesture|expression):([^}]+)\}'
        )

        for match in COMBINED_RE.finditer(text):
            speech = text[last_end:match.start()].strip()
            if speech:
                actions.append({"type": "speak", "value": speech, "tone": current_tone})

            if match.group(1):  # tone match e.g. {gentle}
                current_tone = match.group(1)
            elif match.group(2):  # gesture or expression match
                cmd_type = match.group(2)   # "gesture" or "expression"
                cmd_value = match.group(3)  # e.g. "happy", "sad"
                actions.append({"type": cmd_type, "value": cmd_value})

            last_end = match.end()

        speech = text[last_end:].strip()
        if speech:
            actions.append({"type": "speak", "value": speech, "tone": current_tone})

        return actions

    def extract_response_line(llm_output):
        for line in llm_output.split('\n'):
            if line.startswith('RESPONSE:'):
                return line[len('RESPONSE:'):].strip()
        return llm_output  # fallback: treat whole thing as response

    BLOCKING_GESTURES = {
    "yawn", "drink", "stretching", "breathing_exercise",
    "peekaboo", "touch_head", "bored",
}

    def execute_response(response_text, command_interface):
        response_only = extract_response_line(response_text)
        actions = parse_response(response_only)
        print("Parsed actions:", actions)
        
        pending_nonblocking = None  # holds a gesture/expression to fire alongside next speech

        consumed = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue

            if action["type"] == "speak":
                text = action["value"]
                if action.get("tone") and action["tone"] in TONE_MAP:
                    text = TONE_MAP[action["tone"]] + text
                if pending_nonblocking:
                    t = threading.Thread(target=lambda a=pending_nonblocking: (
                        ci._cmd_perform_gesture({"gesture": a["value"]}) if a["type"] == "gesture"
                        else ci._cmd_perform_emotion({"emotion": a["value"]})
                    ), daemon=True)
                    t.start()
                    pending_nonblocking = None
                ci._cmd_talk({"message": text})

            elif action["type"] == "gesture":
                if action["value"] in BLOCKING_GESTURES:
                    next_action = actions[i + 1] if i + 1 < len(actions) else None
                    if next_action and next_action["type"] == "expression":
                        consumed.add(i + 1)
                        threading.Thread(target=lambda e=next_action: ci._cmd_perform_emotion(
                            {"emotion": e["value"]}), daemon=True).start()
                    ci._cmd_perform_gesture({"gesture": action["value"]})
                else:
                    pending_nonblocking = action

            elif action["type"] == "expression":
                pending_nonblocking = action

        if pending_nonblocking:
            if pending_nonblocking["type"] == "gesture":
                ci._cmd_perform_gesture({"gesture": pending_nonblocking["value"]})
            else:
                ci._cmd_perform_emotion({"emotion": pending_nonblocking["value"]})
    #ci._cmd_talk({'message': "Hello! I am your emotional robot companion. How are you feeling today?"})
    ci._cmd_talk({'message': "Hello"})

    # Pass setup args via setup_kwargs — BaseNode calls setup() and starts the thread automatically
    asr = RivaSpeechRecognitionSilero(setup_kwargs={
        'language': 'en-US',
        'detection_timeout': 30,
        'use_vad': True,
        'event_callback': event_callback,
        'continuous_recog_callback': transcript_callback
    })

    
