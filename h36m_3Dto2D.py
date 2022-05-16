import json
import pathlib
import numpy as np
import cv2
import json 
from collections import OrderedDict
# action id(2~16), subaction id(1,2), frame id(다 다름)
# action 15개 * subaction 2개 *  카메라(4)시점 별 폴더  = 총 폴더는 120개
# 17개의 keypoint에 대한 3d ground truth(=x,y,z)

def world2cam(world_coord,i, R, T):
    cam_coord = np.dot(R,world_coord[i].reshape(3,1)).reshape(1,3) + T.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord,f,c):
    intrinsic = [[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1]]
    pixel_coord = np.dot(intrinsic,cam_coord.reshape(3,1))
    return pixel_coord

def count(j,k,c):#folder내의 frame count
    initial_count = 0 

    for path in pathlib.Path("../../../mmdetection/data/h36m/H36M/s_06/Videos/s_06_act_"+str(j).zfill(2)+"_subact_"+str(k).zfill(2)+"_ca_"+str(c).zfill(2)).iterdir():
        if path.is_file():
            initial_count += 1
    return initial_count

with open('./Human36M_subject6_joint_3d.json','r') as f: # 3d pose ground truth annotation
    json_3d=json.load(f)
with open('./Human36M_subject6_camera.json','r') as c: # 카메라 파라미터 annotation
    json_cam=json.load(c)

action = OrderedDict()
subaction = OrderedDict()
frame = OrderedDict()

#img = cv2.imread("s_08_act_02_subact_01_ca_01_000001.jpg")
#test_num = 1

for j in range(15):#action ( j ) , 2부터 시작
    for k in range(2):#subaction ( k ), 1부터 시작
        for c in range(4):#camera view 
            cam_parm = json_cam[str(c+1)] # j+1번째 카메라의 parameter(R,T,F,C)
            frame_full = count(j+2,k+1,c+1)# total frames
            for i in range(frame_full):#f rame ( i )
                gt_3d = json_3d[str(j+2)][str(k+1)][str(i)]#frame별 3d ground truth, 17개의 (x,y,z) 좌표
                annotations_2d = []
                print("action",j+2," ","subaction",k+1," ","frame",i)
                for key in range(17): # keypoint 17개
                    cam_coord = world2cam(np.array(gt_3d),key,np.array(cam_parm["R"]),np.array(cam_parm["t"]))
                    pixel_coord = cam2pixel(cam_coord,cam_parm["f"],cam_parm["c"])
                    pixel_coord = [float(pixel_coord[0]/pixel_coord[2]), float(pixel_coord[1]/pixel_coord[2])]
                    #if test_num == 1:
                    #    cv2.circle(img,tuple(pixel_coord),5,(0,0,255),5)
                    
                    annotations_2d.append(pixel_coord)
                #if test_num ==1 :
                #    cv2.imwrite("2d GT_final.jpg",img)
                    
                
                frame[i] = annotations_2d
                action[j+2] = subaction
                subaction[k+1] = frame
                #test_num +=1
file_path = "./Human3.6M_subject6_joint_2d_converted.json"

with open( file_path, 'w', encoding = 'utf-8') as file:
    json.dump(action,file)
