from ultralytics import YOLO
import streamlit as st
import cv2
import pafy
import pickle
import settings

# Load pickled model (Ensure path is correct)
with open('C:\\Users\\DELL\\Desktop\\yolov8-waste-classification-streamlitapp-main\\yolov8-waste-classification-streamlitapp-main\\yolov8\\streamlit-detection-tracking - app\\weights\\yolov8 (1).pkl', 'rb') as file:
    model1 = pickle.load(file)

# Load YOLO model from a .pt file
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    """
    model = YOLO('C:\\Users\\DELL\\Desktop\\yolov8-waste-classification-streamlitapp-main\\yolov8-waste-classification-streamlitapp-main\\yolov8\\streamlit-detection-tracking - app\\weights\\yoloooo.pt')
    return model

# Display tracker options in Streamlit sidebar
def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

# Display detected objects on video frames
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (640, 640))  # Resize to the model's expected input size

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Webcam Feed',
                   channels="BGR",
                   use_column_width=True)

# Play YouTube videos and perform detection
def play_youtube_video(conf, model):
    """
    Plays a YouTube video and detects objects in real-time using YOLOv8.
    """
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Trash'):
        try:
            video = pafy.new(source_youtube)
            best = video.getbest(preftype="mp4")
            vid_cap = cv2.VideoCapture(best.url)
            st_frame = st.empty()

            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# Play webcam video feed and detect objects
def play_webcam(conf, model):
    """
    Plays a webcam stream and detects objects in real-time using YOLOv8.
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Trash'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()

            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    # Image preprocessing
                    image = cv2.resize(image, (640, 640))  # Resize to the model's input size
                    image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian Blur

                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading webcam feed: " + str(e))

# Play stored video and perform detection
def play_stored_video(conf, model):
    """
    Plays a stored video and detects objects in real-time using YOLOv8.
    """
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Trash'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()

            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    # Resize and process image
                    image = cv2.resize(image, (640, 640))  # Resize to the model's input size
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# Main Streamlit interface
def main():
    st.title("YOLOv8 Waste Detection and Tracking")

    # Load the YOLO model
    model = load_model('path_to_your_model_file.pt')  # Adjust to your model path

    # Sidebar options for confidence threshold
    conf = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.5)

    # Options to choose the video source
    option = st.sidebar.selectbox('Choose Video Source', ['YouTube', 'Webcam', 'Stored Video'])

    if option == 'YouTube':
        play_youtube_video(conf, model)
    elif option == 'Webcam':
        play_webcam(conf, model)
    elif option == 'Stored Video':
        play_stored_video(conf, model)

if __name__ == "__main__":
    main()
