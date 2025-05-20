import torch
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from collections import deque 
from inference import get_model
from transformers import AutoProcessor, SiglipVisionModel
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_paths_on_pitch
from teamClassifier import TeamClassifier, resolve_goalkeepers_team_id, replace_outliers_based_on_distance


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


ROBOFLOW_API_KEY ="19956OGitmg91nRKHH4L"
PLAYER_DETECTION_MODEL_ID="football-players-detection-3zvbc/11"
PLAYER_MODEL_DETECTION=get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

FIELD_DETECTION_MODEL_ID="football-field-detection-f07vi/15"
FIELD_MODEL_DETECTION=get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)


CONFIG= SoccerPitchConfiguration()

SOURCE_VIDEO_PATH="./121364_0.mp4"
TAGET_VIDEO_PATH="./121364_0_result_1.mp4"
TAGET_VIDEO_PATH_PITCH='./121364_0_result_2.mp4'
TAGET_VIDEO_PATH_ZONE='./121364_0_result_3.mp4'
TAGET_VIDEO_PATH_BALL='./121364_0_result_4.mp4'


STRIDE= 30
BATCH_SIZE = 32

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3



ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
)



frame_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result = PLAYER_MODEL_DETECTION.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device="cuda")
team_classifier.fit(crops)



# print("Empieza primer video")
# #Video1
# video_info= sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# video_sink=sv.VideoSink(TAGET_VIDEO_PATH, video_info=video_info)
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

tracker = sv.ByteTrack()
tracker.reset()

# with video_sink:
#     for frame in tqdm(frame_generator, total=video_info.total_frames):
#         result= PLAYER_MODEL_DETECTION.infer(frame, confidence=0.3, device=device)[0]
#         detections= sv.Detections.from_inference(result)

#         ball_detections = detections[detections.class_id == BALL_ID]
#         ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
#         all_detections = detections[detections.class_id != BALL_ID]
#         all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
#         all_detections = tracker.update_with_detections(detections=all_detections)
        
#         goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#         players_detections = all_detections[all_detections.class_id == PLAYER_ID]
#         referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

#         players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
#         players_detections.class_id = team_classifier.predict(players_crops)

#         goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
#             players_detections, goalkeepers_detections)

#         referees_detections.class_id -= 1

#         all_detections = sv.Detections.merge([
#             players_detections, goalkeepers_detections, referees_detections])

#         labels = [
#             f"#{tracker_id}"
#             for tracker_id
#             in all_detections.tracker_id
#         ]

#         all_detections.class_id = all_detections.class_id.astype(int)

#         annotated_frame = frame.copy()
#         annotated_frame = ellipse_annotator.annotate(
#             scene=annotated_frame,
#             detections=all_detections)
#         annotated_frame = label_annotator.annotate(
#             scene=annotated_frame,
#             detections=all_detections,
#             labels=labels)
#         annotated_frame = triangle_annotator.annotate(
#             scene=annotated_frame,
#             detections=ball_detections)
        
#         video_sink.write_frame(annotated_frame)
# print("Primer video exportado")        


# print("Empieza segundo video")
# #Video2
# video_info= sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# video_sink = sv.VideoSink( TAGET_VIDEO_PATH_PITCH, video_info=video_info)
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)


# with video_sink:
#     for frame in tqdm(frame_generator, total=video_info.total_frames):
#         result = PLAYER_MODEL_DETECTION.infer(frame, confidence=0.3, device=device)[0]
#         detections = sv.Detections.from_inference(result)

#         ball_detections = detections[detections.class_id == BALL_ID]
#         ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

#         all_detections = detections[detections.class_id != BALL_ID]
#         all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
#         all_detections = tracker.update_with_detections(detections=all_detections)

#         goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#         players_detections = all_detections[all_detections.class_id == PLAYER_ID]
#         referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

#         players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
#         players_detections.class_id = team_classifier.predict(players_crops)

#         goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
#             players_detections, goalkeepers_detections)

#         referees_detections.class_id -= 1

#         all_detections = sv.Detections.merge([
#             players_detections, goalkeepers_detections, referees_detections])


#         result= FIELD_MODEL_DETECTION.infer(frame, confidence=0.3)[0]
#         key_points= sv.KeyPoints.from_inference(result)

#         filter= key_points.confidence[0]>0.5
#         frame_reference_points= key_points.xy[0][filter]
#         frame_reference_key_points=sv.KeyPoints(xy=frame_reference_points[np.newaxis,...])
#         pitch_reference_points=np.array(CONFIG.vertices)[filter]

#         view_transformer= ViewTransformer(
#             source=frame_reference_points,
#             target=pitch_reference_points
#         )

#         frame_ball_xy= ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_ball_xy= view_transformer.transform_points(frame_ball_xy)

#         frame_players_xy= players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_players_xy= view_transformer.transform_points(frame_players_xy)
        
#         frame_goalkeepers_xy= goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_goalkeepers_xy= view_transformer.transform_points(frame_goalkeepers_xy)

#         frame_referees_xy= referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_referees_xy= view_transformer.transform_points(frame_referees_xy)


#         pitch= draw_pitch(config=CONFIG)
#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_ball_xy,
#             face_color=sv.Color.WHITE,
#             edge_color=sv.Color.BLACK,
#             radius=10,
#             pitch=pitch
#         )

#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_players_xy[players_detections.class_id==0],
#             face_color=sv.Color.from_hex("#00BFFF"),
#             edge_color=sv.Color.BLACK,
#             radius=10,
#             pitch=pitch
#         )
        
#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_goalkeepers_xy[goalkeepers_detections.class_id==0],
#             face_color=sv.Color.from_hex("#00BFFF"),
#             edge_color=sv.Color.BLACK,
#             radius=10,
#             pitch=pitch
#         )

#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_players_xy[players_detections.class_id==1],
#             face_color=sv.Color.from_hex("#FF1493"),
#             edge_color=sv.Color.BLACK,
#             radius=10,
#             pitch=pitch
#         )
        
#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_goalkeepers_xy[goalkeepers_detections.class_id==1],
#             face_color=sv.Color.from_hex("#00BFFF"),
#             edge_color=sv.Color.BLACK,
#             radius=10,
#             pitch=pitch
#         )

#         pitch= draw_points_on_pitch(
#             config=CONFIG,
#             xy= pitch_referees_xy,
#             face_color=sv.Color.from_hex("#FFD700"),
#             edge_color=sv.Color.BLACK,
        
#             radius=10,
#             pitch=pitch
#         )
        
#         # Redimensionar si es necesario
#         if (pitch.shape[1], pitch.shape[0]) != (video_info.width, video_info.height):
#             pitch = cv2.resize(pitch, (video_info.width, video_info.height))

#         video_sink.write_frame(pitch)
# print("Segundo video exportado")  
    

# print("Empieza tercer video")
# #Video3
# video_info= sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# video_sink = sv.VideoSink(TAGET_VIDEO_PATH_ZONE, video_info=video_info)
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# with video_sink:
#     for frame in tqdm(frame_generator, total=video_info.total_frames):
#         result = PLAYER_MODEL_DETECTION.infer(frame, confidence=0.3, device=device)[0]
#         detections = sv.Detections.from_inference(result)

#         ball_detections = detections[detections.class_id == BALL_ID]
#         ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

#         all_detections = detections[detections.class_id != BALL_ID]
#         all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
#         all_detections = tracker.update_with_detections(detections=all_detections)

#         goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#         players_detections = all_detections[all_detections.class_id == PLAYER_ID]
#         referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

#         players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
#         players_detections.class_id = team_classifier.predict(players_crops)

#         goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
#             players_detections, goalkeepers_detections)

  
#         result= FIELD_MODEL_DETECTION.infer(frame, confidence=0.3)[0]
#         key_points= sv.KeyPoints.from_inference(result)

#         filter= key_points.confidence[0]>0.5
#         frame_reference_points= key_points.xy[0][filter]
#         frame_reference_key_points=sv.KeyPoints(xy=frame_reference_points[np.newaxis,...])
#         pitch_reference_points=np.array(CONFIG.vertices)[filter]

#         view_transformer= ViewTransformer(
#             source=frame_reference_points,
#             target=pitch_reference_points
#         )

#         frame_ball_xy= ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_ball_xy= view_transformer.transform_points(frame_ball_xy)

#         frame_players_xy= players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_players_xy= view_transformer.transform_points(frame_players_xy)
        
#         frame_goalkeepers_xy= goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_goalkeepers_xy= view_transformer.transform_points(frame_goalkeepers_xy)

#         frame_referees_xy= referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         pitch_referees_xy= view_transformer.transform_points(frame_referees_xy)

#         pitch= draw_pitch(config=CONFIG)
        
#         pitch=draw_pitch_voronoi_diagram(
#             config=CONFIG,
#             team_1_xy=pitch_players_xy[players_detections.class_id==0],
#             team_2_xy=pitch_players_xy[players_detections.class_id==1],
#             team_1_color=sv.Color.from_hex('#00BFFF'),
#             team_2_color=sv.Color.from_hex('#FF1493'),
#             pitch=pitch
#         )

#         # Redimensionar si es necesario
#         if (pitch.shape[1], pitch.shape[0]) != (video_info.width, video_info.height):
#             pitch = cv2.resize(pitch, (video_info.width, video_info.height))

#         video_sink.write_frame(pitch)
# print("Tercer video exportado")         
  
print("Empieza cuarto video")      
#Video4
video_info= sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TAGET_VIDEO_PATH_BALL, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

MAXLEN=5
path_raw=[]
M= deque(maxlen=MAXLEN)
MAX_DISTANCE_THRESHOLD = 500

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = PLAYER_MODEL_DETECTION.infer(frame, confidence=0.3, device=device)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
  
        result= FIELD_MODEL_DETECTION.infer(frame, confidence=0.3)[0]
        key_points= sv.KeyPoints.from_inference(result)

        filter= key_points.confidence[0]>0.5
        frame_reference_points= key_points.xy[0][filter]
        frame_reference_key_points=sv.KeyPoints(xy=frame_reference_points[np.newaxis,...])
        pitch_reference_points=np.array(CONFIG.vertices)[filter]

        view_transformer= ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        
        M.append(view_transformer.m)
        view_transformer.m=np.mean(np.array(M), axis=0)

        frame_ball_xy= ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy= view_transformer.transform_points(frame_ball_xy)
        
        for coorinates in pitch_ball_xy:
            if coorinates.shape[0] >= 2:
                path_raw.append(np.empty((0, 2), dtype=np.float32).flatten() )
            else:
                path_raw.append(coorinates.flatten())

        pitch= draw_pitch(config=CONFIG)
        
        path = replace_outliers_based_on_distance(path_raw, MAX_DISTANCE_THRESHOLD)
        
        pitch= draw_paths_on_pitch(
            config=CONFIG,
            paths=[path],
            color=sv.Color.WHITE,
            pitch=pitch)
        
        print(pitch)

        # Redimensionar si es necesario
        if (pitch.shape[1], pitch.shape[0]) != (video_info.width, video_info.height):
            pitch = cv2.resize(pitch, (video_info.width, video_info.height))

        video_sink.write_frame(pitch)
print("Cuarto video exportado")  