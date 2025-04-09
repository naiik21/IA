import supervision as sv
import cv2
import numpy as np
from inference import get_model
from ultralytics import YOLO

# ROBOFLOW_API_KEY ="zBybZbMzBSUu4cAR3U7F"
# MODEL_ID = "basketball-detection-1mtj3/1"
# pre_model= get_model( model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)

CLASSES=[7]

model = YOLO('yolov8m.pt')
# print(model.names)

tracker= sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)


def main(video_path):
    # Generador de cuadros del video
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    
    for frame in frame_generator:
   
        # Redimensiona el cuadro        
        frame=cv2.resize(frame, (1000, 500))
        # Realiza la predicción en el cuadro actual
        results = model(frame, device='cuda', verbose=False)[0]
        # Renderiza las detecciones en el cuadro
        detections = sv.Detections.from_ultralytics(results)
        # detections = detections[np.isin(detections.class_id, CLASSES)]
        detections=tracker.update_with_detections(detections)
        
        labels =[
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]
        
        frame_annotated = box_annotator.annotate(scene=frame, detections=detections)
        frame_annotated = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Muestra el cuadro procesado
        #cv2.imshow('Detections', frame_annotated)
        
        # Salir con 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    
    # Libera los recursos
    #cv2.destroyAllWindows()

main('./jaM.mp4')



