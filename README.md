# Face-Recognition-OpenCV-and-deep-learning

<img width="500" alt="portfolio_view" src="https://github.com/ashish1sasmal/Face-Recognition-OpenCV-and-deep-learning/blob/master/snape_detect.jpg">
<img width="500" alt="portfolio_view" src="https://github.com/ashish1sasmal/Face-Recognition-OpenCV-and-deep-learning/blob/master/Screenshot%20from%202020-07-14%2005-57-46.png">

## Tools and Technologies used in this project
####   1. dlib
####   2. face_recognition
####   3. cv2 (opencv-python)
####   4. Microsoft’s Bing Image Search API. https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/
####   5. CNN/HOG face recognizer (Deep Learning)
####   6. numpy (ofcourse)

#### Files Structure:
        ├── dataset
        │   ├── Entries [n entries]
        ├── examples
        │   ├── example_01.png
        │   ├── example_02.png
        │   └── example_03.png
        ├── output
        │   └── lunch_scene_output.avi
        ├── videos
        │   └── lunch_scene.mp4
        ├── search_bing_api.py
        ├── encode_faces.py
        ├── recognize_faces_image.py
        ├── recognize_faces_video.py
        ├── recognize_faces_video_file.py
        └── encodings.pickle


### Important Performance Note:
The CNN face recognizer should only be used in real-time if you are working with a GPU 
(you can use it with a CPU, but expect less than 0.5 FPS which makes for a choppy video). 
Alternatively (you are using a CPU), you should use the HoG method
(or even OpenCV Haar cascades covered in a future blog post) and expect adequate speeds. 

