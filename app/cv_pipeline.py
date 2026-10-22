import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_video_real(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # give the frame to the model for prediction
        results = model.predict(source=frame, conf=0.5, verbose=False)
        
        annotated_frame = results[0].plot()
        
        # save frame to output video
        out.write(annotated_frame)
        frames_processed += 1
        
    cap.release()
    out.release()
    
    return {"frames_processed": frames_processed, "model_used": "YOLO"}