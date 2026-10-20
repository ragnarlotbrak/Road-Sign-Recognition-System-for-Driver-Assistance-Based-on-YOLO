import cv2

def process_video_dummy(input_path: str, output_path: str):
    # Открываем видеофайл
    cap = cv2.VideoCapture(input_path)
    
    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Настраиваем инструмент для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Видео закончилось
            
        # --- for further yolo ml ---
        # temp
        start_point = (100, 100) 
        end_point = (300, 300)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        
        # text
        cv2.putText(frame, "Speed Limit 60", (100, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        # ------------------------------------
        
        # saving the processed frame to the output video
        out.write(frame)
        frames_processed += 1
        
    cap.release()
    out.release()
    
    return {"frames_processed": frames_processed, "mock_detections": True}