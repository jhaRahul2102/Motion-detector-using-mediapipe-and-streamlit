

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import cv2
from streamlit_lottie import st_lottie
import json
import av
import streamlit as st
import numpy as np


def load_lottiefile(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)




# Setting Page Configuration:-

st.set_page_config(page_title='Hand motion detection using mediapie',layout='centered',initial_sidebar_state='expanded')





mp_hands=mp.solutions.hands
drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class Faceemotion(VideoTransformerBase):
  


  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    image=cv2.flip(img,1)
    height,width,_=image.shape
    results=hands.process(image)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        drawing.draw_landmarks(image=image, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=drawing.DrawingSpec(color=(255,255,255),thickness=2, circle_radius=2),
                                      connection_drawing_spec=drawing.DrawingSpec(color=(0,255,0),thickness=2, circle_radius=2))
      
      count = {'RIGHT': 0, 'LEFT': 0} 
      finger_tips_id=[mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
      fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,   # Status of finger if not moved.
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}       

      
      cv2.putText(image,'Hand type',(height-20,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,245, 250), 3,cv2.LINE_AA) 

      for hand_info in results.multi_handedness:
        if(hand_info.classification[0].label=='Right'):
          cv2.putText(image,hand_info.classification[0].label,(height-20,80), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3,cv2.LINE_AA) 
        elif(hand_info.classification[0].label=='Left'):
          cv2.putText(image,hand_info.classification[0].label,(height-20,80), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 3,cv2.LINE_AA) 
        
          




      for hand_index, hand_info in enumerate(results.multi_handedness):    
        hand_label = hand_info.classification[0].label
        hand_landmarks =  results.multi_hand_landmarks[hand_index] 
        
              
        for tip_index in finger_tips_id:                 
          finger_name = tip_index.name.split("_")[0]    # Extract name of the finger
            
          if(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y): 
            fingers_statuses[hand_label.upper()+"_"+finger_name] = True
            count[hand_label.upper()] += 1
      thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
      thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        
      if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
        fingers_statuses[hand_label.upper()+"_THUMB"] = True
        count[hand_label.upper()] += 1 

      cv2.putText(image,str(sum(count.values())),(20,width-200), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,250,0), 3,cv2.LINE_AA)
       
      
    cv2.putText(image, 'Number of finger:-',(0,width-250), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0,255,200), 2,cv2.LINE_AA)
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")




def main():
  st.title('Hand Motion detector using Mediapipe',)
  activit= ["About the project", "Working of the project", "About me"]
  choice = st.sidebar.selectbox("Choose the option", activit)
  if(choice=='About the project'):
    st.text('')
    st.text('')
    st.text('')
    st.header('About mediapipe:-')
    st.markdown("""MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame. Whereas current state-of-the-art approaches rely primarily on powerful desktop environments for inference, our method achieves real-time performance on a mobile phone, and even scales to multiple hands. We hope that providing this hand perception functionality to the wider research and development
     community will result in an emergence of creative use cases, stimulating new applications and new research avenues.""")
    st.markdown('**To run the project please choose from activity from side menu.**')
    st.text('-------------------------------------------------------------------------')
    st.text('')
    st.text('')
    st.text('')
    st.subheader('Working of mediapipe')
    st.image('/content/hand_tracking_3d_android_gpu.gif')
    st.markdown("""MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. This strategy is similar to that employed in our MediaPipe Face Mesh solution, which uses a face detector together with a face landmark model.
     Providing the accurately cropped hand image to the hand landmark model drastically reduces the need for data augmentation (e.g. rotations, translation and scale) and instead allows the network to dedicate most of its capacity towards coordinate prediction accuracy. In addition, in our pipeline the crops can also be generated based on the hand landmarks identified in the previous frame, and only when the landmark model could no longer identify hand presence is palm detection invoked to relocalize the hand.""")
    st.text('-------------------------------------------------------------------------')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.header('Working of finger counter:-')
    st.image('/content/hand_landmarks.png')
    st.markdown("""For finger counting we run a condition statement where if tip of finger is lower than middle finger then we take into consideration and for thumb we 
    compare the x value of hand.""")
    st.text('--------------------------------------------------------------------------')
    st.text('')
    st.text('')
    st.text(' ')
    st.header('Resource:-')
    st.markdown('1. **https://google.github.io/mediapipe/solutions/hands.**')
    st.markdown('2. **https://www.youtube.com/watch?v=Af5Y9bBLA7s**')
  
  elif(choice=='Working of the project'):
    

    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion,async_processing=True)
    
    st.text('Click on the button to start the video and please give permission to camera.')
    st.text('---------------------------------------------------------------------------')
    st.text('')
    st.text('')
    st.text(' ')
    st.subheader('Aspect of project:-')
    st.markdown('1.If you place your hand in front of your camera then it will identify the motion of the hand.')
    st.markdown('2.It identify right or left hand in motion.')
    st.markdown('3.If you show finger then it will coun number of finger.')
    st.text('')
    st.text('')
    st.text(' ')
    st.text('---------------------------------------------------------------------------')
    st.text('')
    st.text(' ')
    st.subheader('Disclaimer:-')
    st.markdown('1.:-Please place yur hand in front and wait for few second to showcase.')
    st.markdown('2.Frame rate can be slow depending on the connection.')
  elif(choice=='About me'):
     st.header('About me')
     lottie_coding2=load_lottiefile("80680-online-study.json")
     st_lottie(lottie_coding2,speed=1,reverse=False,loop=True,quality="low", height=500,width=500,key=None)
     st.markdown("""**I have recently graduate in 2021 and was not able to get a job in campus placement.So for last 5 months I have being reworking for my passion in data science.I am currently doing internship upder Almabetter.**""")
     st.text(' ')
     st.text('  ')
     st.markdown('**A chance  would be appreciated....**')
     st.text(' ')
     st.subheader('Contact:-')
     st.markdown('**Phone ☎️ :- 6202239544**')
     st.markdown('**Email 📧 :- rahuljha0610@gmail.com**')
     st.markdown('**Linkedin 🚦 :- [link](https://www.linkedin.com/in/rahul-jha-600047164/)**')



if __name__=='__main__':
  main()