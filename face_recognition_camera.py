from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
#from live_cam import live

cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
# load the input image and convert it from BGR to RGB

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

while True:
    # image = live()
    image= vs.read()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = imutils.resize(image, width=750)
    r = image.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,
    	model=args["detection_method"])

    encodings = face_recognition.face_encodings(rgb, boxes)
    # initialize the list of names for each face detected
    names = []


    # loop over the facial embeddings
    for encoding in encodings:
    	# attempt to match each face in the input image to our known
    	# encodings
    	matches = face_recognition.compare_faces(data["encodings"],
    		encoding)
    	name = "Unknown"

    	# check to see if we have found a match
    	if True in matches:
    		# find the indexes of all matched faces then initialize a
    		# dictionary to count the total number of times each face
    		# was matched
    		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    		counts = {}
    		# loop over the matched indexes and maintain a count for
    		# each recognized face face
    		for i in matchedIdxs:
    			name = data["names"][i]
    			counts[name] = counts.get(name, 0) + 1
    		# determine the recognized face with the largest number of
    		# votes (note: in the event of an unlikely tie Python will
    		# select first entry in the dictionary)
    		name = max(counts, key=counts.get)

    	# update the list of names
    	names.append(name)



    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):

        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

    	# if the video writer is None *AND* we are supposed to write
    	# the output video to disk initialize the writer
        # if writer is None and args["output"] is not None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(args["output"], fourcc, 20,
    	# 		(image.shape[1], image.shape[0]), True)
    	# # if the writer is not None, write the frame with recognized
    	# # faces to disk
        # if writer is not None:
        #     writer.write(image)


        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()

if writer is not None:
	writer.release()
