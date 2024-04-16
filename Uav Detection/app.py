from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
import ultralytics

app = Flask(__name__)

# Define global variables for the model and its parameters
classes = {1: 'DRONE'}
model = None
processing_status = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
selected_model=" "
i=1

# Define paths for different models
model_paths = {
    "model1": 'uploads/uav_trained_faster_rcnn.pth',  # Path for Faster RCNN
    "model2": 'uploads/uav_trained_ssd.pth',  # Path for Single Shot Decoder
    "model3": 'uploads/best (2).pt'   # Path for Model 3
}

# Load model parameters and set model to evaluation mode
def load_model_parameters(model_path):
    global model
    # Load model
    if torch.cuda.is_available():
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    #model = torch.load(model_path)
    try:
        model.to(device)
        model.eval()
    except Exception as e:
        print("Error loading model parameters:", e)

# Load the third model
model3_path = 'uploads/uav_trained_yolo.pt'
model3 = None  # Define global variable for the third model

def load_model3_parameters(model_path):
    global model3
    try:
        model3 = ultralytics.YOLO(model_path) 
        class_map ={ 0 : 'DRONE'}
    except Exception as e:
        print("Error loading model parameters for Model 3:", e)

#load_model3_parameters(model3_path)

# Load the default model
default_model = "model3"  # Set default model
#load_model_parameters(model_paths[default_model])

def visualize_detections(image, results, class_map):
    colors = np.random.uniform(0, 255, size=(len(class_map), 3))  # Generate random colors for classes
    for detections in results:  # Iterate over each detection
        if len(detections.boxes) == 0:  # Check if there are no detections
            continue  # Skip this iteration if there are no detections

        boxes = detections.boxes[0]  # Extract bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()[:4]  # Extract bounding box coordinates
            conf, cls = box.conf.item(), box.cls.item()  # Confidence score and class index
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{class_map[int(cls)]}: {conf:.2f}"
            color = colors[int(cls)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 0, 0), 2)
    return image

# Function to detect objects using Model 3
def obj_detector_model3(img):
    image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    #new_image = cv2.resize(image, (imgsz, imgsz))
    new_image=image
    # Perform object detection using YOLOv8 model
    results = model3(new_image)

    # Visualize detections on the image
    class_map ={ 0 : 'DRONE'}
    annotated_image = visualize_detections(img, results, class_map)

    return annotated_image

# Function to detect objects in an image
def obj_detector(img):
    global selected_model
    if selected_model == 'model3':
        return obj_detector_model3(img)
    else:
        img1=img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0,3,1,2)

        model.eval()
        if selected_model == 'model1':
            detection_threshold = 0.95
        else:
            detection_threshold = 0.7
        img = list(im.to(device) for im in img)
        output = model(img)

        boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        for i, box in enumerate(boxes):
            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
            cv2.putText(img1, classes[labels[i]] + " " + str(scores[i]), (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 2, cv2.LINE_AA)

        return img1

@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_model
    if request.method == 'POST':
        selected_model = request.form['model']
        if selected_model == 'model3':
            load_model3_parameters(model_paths[selected_model])
        else:
            load_model_parameters(model_paths[selected_model])
        if 'file' in request.files:
            file = request.files['file']
            if file:
                # Check if the uploaded file is an image or a video
                file_extension = file.filename.split('.')[-1].lower()
                if file_extension in ['jpg', 'jpeg', 'png', 'gif']:  # Assuming these are image extensions
                    # Read the uploaded image
                    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                    # Process the image
                    processed_image = obj_detector(image)
                    # Save the processed image
                    output_path = 'static/output_image.jpg'
                    cv2.imwrite(output_path, processed_image)
                    processing_status = "Image processed successfully"
                    return render_template('detection.html', image_url=output_path, processing_status=processing_status)
                elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:  # Assuming these are video extensions
                    # Save the uploaded video file
                    i=0
                    video_path = 'uploads/' + file.filename
                    file.save(video_path)
                    # Process the video file
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the input video
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if ((i%5) ==0):
                            processed_frame = obj_detector(frame)
                            i+=1
                        else:
                            processed_frame=frame
                            i+=1
                        frames.append(processed_frame)
                    cap.release()
                    # Write the processed frames to a new video file
                    out_path = 'static/output.avi'
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frames[0].shape[1], frames[0].shape[0]))
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    processing_status = "Video processed successfully"
                    return render_template('detection.html', video_url=out_path, processing_status=processing_status)

    processing_status = None
    return render_template('index.html', processing_status=processing_status)

@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.form['model']
    if selected_model == 'model3':
        load_model3_parameters(model_paths[selected_model])
    else:
        load_model_parameters(model_paths[selected_model])
    return "model selected success"

if __name__ == '__main__':
    app.run(debug=True)
