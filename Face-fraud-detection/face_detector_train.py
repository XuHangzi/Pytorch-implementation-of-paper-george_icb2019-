import numpy as np
import cv2
import dlib
import os
paths = os.walk(r'./train_release')
for path, dir_lst, file_lst in paths:
    for file_name in file_lst:
        print(file_name)
        #print(os.path.join(path, file_name))
        video_path=os.path.join(path, file_name)
        #记录读取的帧数
        times=0
        #记录保存图片数
        num=1
        #提取视频的频率，每十帧提取一个
        if file_name == '1.avi' or file_name =='2.avi' or file_name =='HR_1.avi':
            frameFrequency=3
        else:
            frameFrequency=10
        #记录保存图片的目录
        #输出图片到当前目录文件夹下
        outPutDirName=path.replace('train_release', 'video/train')
        outPutDirName=outPutDirName+'/'+file_name+'/'
        outPutDirName=outPutDirName.replace('.avi', '')
        if not os.path.exists(outPutDirName):
            os.makedirs(outPutDirName)
        cap=cv2.VideoCapture(video_path)
        
        
        while True:
            times+=1
            res,img = cap.read()
            if not res:
                print('video ends')
                assert img is None
                break
            if times%frameFrequency==0:
                #dlib_detector_face(image)
                detector = dlib.get_frontal_face_detector()
                # 1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
                dets = detector(img, 1)
                #print('dets:', dets)  # dets: rectangles[[(118, 139) (304, 325)]]
                #print("Number of faces detected: {}".format(len(dets)))
                for i, d in enumerate(dets):
                    #因为有个别视频里有人走动，会导致锁定到多个人脸，因此只取第一个人脸
                    if i==0:
                    # 人脸的左上和右下角坐标
                        left = d.left()
                        top = d.top()
                        right = d.right()
                        bottom = d.bottom()
                        p=bottom-top
                        img=np.pad(img,((p,p),(p,p),(0,0)),'edge')
                        left+=p;top+=p;right+=p;bottom+=p
                    #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))
                        cv2.rectangle(img, (left, top), (right, bottom), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        new_img=img[top:bottom,left:right]
                        new_img=cv2.resize(new_img,(224,224))
                        image_path = outPutDirName + str(num)+'.jpg'
                    #image_path = r'./vedio/train/{}/{}.jpg'.format(str(f),str(num))
                        retval=cv2.imwrite(image_path, new_img)
                        num+=1 #每截出一张人脸，图的数量就加一
                    #print('num:'+str(num)+',frame:'+str(times))
                #print(outPutDirName+str(times)+'.jdp')#说明这张图是第几帧
            cv2.waitKey(1)
        print('图片提取结束')
        cap.release()