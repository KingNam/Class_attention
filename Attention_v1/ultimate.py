# 눈 면적 구하려는 시도 중

from turtle import bgcolor
import cv2 as cv
# from git import Head
import mediapipe as mp
import time
import utils, math
import numpy as np

# constants
FONTS =cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 
           251, 389, 356, 454, 323, 361, 
           288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 
           150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 
      17, 314, 405, 321, 375,
      291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 
      95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 
      310, 311, 312, 13, 82, 81, 42, 183, 78 ]

LOWER_LIPS =[61, 146, 91, 181, 84, 
             17, 314, 405, 321, 375, 291, 
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

UPPER_LIPS=[ 185, 40, 39, 37,0 ,
            267 ,269 ,270 ,409, 415, 310, 
            311, 312, 13, 82, 81, 42, 183, 78] 

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374,
           373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300,
               276, 283, 282, 295, 285 ]
LEFT_EYE_CENTER = 473

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145,
           153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107,
               55, 65, 52, 53, 46 ]
RIGHT_EYE_CENTER = 468

map_face_mesh = mp.solutions.face_mesh

# Head-Pose indices (NOSE, else)
# 얼굴 3D 모핑을 위한 좌표 
HEAD_FOR = [1,33,263,61,291,199]

# camera object 
camera = cv.VideoCapture(0)

# landmark detection function
# 2차원 점과 3차원 점을 포괄할 수 있게 수정함 
def landmarksDetection(img, results, dimension=2, draw=False):
    
    img_h, img_w, _ = img.shape    
    if dimension == 2:
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_w), int(point.y * img_h)) 
                      for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    
    elif dimension == 3:
        mesh_coord = [(int(point.x * img_w), int(point.y * img_h), int(point.z)) 
                      for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
            
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
# 차후에 코드 수정
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)
    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2
    return ratio 

def HpeEstimate(img, landmarks, index):
    # Usage example: HpeEstimate(img, mesh_coord, HEAD_FOR)
    # landmarks 는 튜플 좌표를 리스트로 가짐
    # 이미지의 높이, 너비, 채널 수 가져옴
    img_h, img_w, _ = img.shape
    
    # 얼굴의 3D/2D face 값 저장할 리스트
    face_3d = []
    face_2d = []
    
    for i in index:
        # 코 landmark를 기준으로 코 좌표 생성
        x = landmarks[i-1][0]
        y = landmarks[i-1][1]
        z = landmarks[i-1][2]
        
        if i == 1:
            nose_2d = (x*img_w, y*img_h)
            nose_3d = (x*img_w, y*img_h, z*3000)
            
        else:
            face_2d.append([x,y])
            face_3d.append([x,y,z])
            
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # 여기서 부터,       
    # The camera matrix
    focal_length = 1 * img_w
    
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    _, rot_vec, _ = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, _ = cv.Rodrigues(rot_vec)
    # 여기까지는 무지성 복사

    # Get angles
    angles,_,_,_,_,_ = cv.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x_ang = angles[0] * 360
    y_ang = angles[1] * 360
    z_ang = angles[2] * 360
    
    nose_p1 = (int(nose_2d[0]), int(nose_2d[1]))
    nose_p2 = (int(nose_2d[0] + y_ang * 10) , int(nose_2d[1] - x_ang * 10))
    
    # 일정 박스 부분을 넘어가면 "cheat" 라고 감지
    if (y_ang > 2) or (y_ang < -2):
        text = "cheat"
    elif (x_ang > 2) or (x_ang < -2):
        text = "cheat"
    else:
        text = "Forward"

    # Display the nose direction
    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    
    return x_ang, y_ang, text, nose_p1, nose_p2

'''
기능 구현
'''

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    '''
    Attention Variable (global)
    '''
    score = 100

    '''
    Running_time Variable
    '''
    total_time = 0
    
    '''
    eye_blink Variable
    '''
    # starting time here 
    eye_closed = False
    Blink_Count = 0
    TOTAL_BLINKS = 0
    CLOSE_TIME = 0
    CLOSING_Blink_CountER = 0
    TEN_CLOSED = 0
    Blink_Standard = 3
    
    '''
    HPE Variable
    '''

    while True:
        start_time = time.time()
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_coords_2d = landmarksDetection(frame, results, dimension=2, draw = False)
            mesh_coords_3d = landmarksDetection(frame, results, dimension=3, draw = False)
            ratio = blinkRatio(frame, mesh_coords_2d, RIGHT_EYE, LEFT_EYE)
            x, y, text, nose_p1, nose_p2 = HpeEstimate(frame, mesh_coords_3d, HEAD_FOR)
            # utils.colorBackgroundText(frame,  text, FONTS, 0.7, (50,100),2, utils.PINK, utils.YELLOW)
                     
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2),text}, direction:{x,y} ', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)
            
            # 개인사용자별로 Ratio 꼭 꼭 꼭 바꿔줘야함
            # 비율이 높을수록 눈을 감은거
            if ratio > 3.5:
                eye_closed = True
                Blink_Count += 1
                utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, 
                                          (int(frame_height/2), 100), 2, utils.YELLOW,
                                          pad_x=6,pad_y=6 )
            
            elif Blink_Count>Blink_Standard:
                TOTAL_BLINKS +=1
                Blink_Count =0  
                
            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            
            cv.polylines(frame,  [np.array([mesh_coords_2d[p] for p in LEFT_EYE ], 
                                           dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords_2d[p] for p in RIGHT_EYE ], 
                                           dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            
         # calculating  frame per seconds FPS
        end_time = time.time()
        total_time += end_time - start_time
        fps = 1/total_time
            
        
        utils.colorBackgroundText(frame,  f'Closed Time, Blink_Counts: {round(CLOSE_TIME,1),TEN_CLOSED}', FONTS, 0.7, (30,150),2,textColor=utils.MAGENTA)  
        utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,180),2,textColor=utils.MAGENTA)  
        utils.colorBackgroundText(frame,  f'Score: {score}', FONTS, 1.0, (800,25),2)  
        # utils.colorBackgroundText(frame,  f'Blink_Count 10s Closed: {TEN_CLOSED}', FONTS, 0.7, (30,270),2) 
        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 25), bgOpacity=0.9, textThickness=2)
        frame =utils.textWithBackground(frame,f'TIME: {round(total_time,1)}',FONTS, 1.0, (30, 55), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_Blink_Counter}.png', frame)
        cv.imshow('frame', frame)
        
        key = cv.waitKey(5)
        
        if cv.waitKey(5) & 0xFF == 27:
            break
        
        
    cv.destroyAllWindows()
    camera.release()