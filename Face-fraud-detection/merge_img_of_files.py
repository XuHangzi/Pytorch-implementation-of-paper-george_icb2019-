import os
import shutil
def merge_img_of_true(data_path,target_path):
    if not os.path.exists(target_path):     #目标文件夹不存在就新建
        os.makedirs(target_path)

    num=1
    for pern in os.listdir(data_path):  #data_path路径下存在你的所有人的文件
        print(pern) #此时打印出来的是你data_path下的所有人
    	#这还没有进入一级目录，我们通过这一步仅仅获取了一级目录的文件名字
    	#那么我们把路径拼接起来，就可以得到第一级目录的路径名啦
        pern_path = data_path + '/' + pern 
    	#这里最重要的是别忘记加这个‘/’啦
    	#因为不加的话就不是一个完整的路径，你可以尝试print(data_path + year)看看效果
        #print(pern_path)#此时打印出来的就是一级目录的路径
    	#现在我们获取了一级目录：pern_path
    	#那么只用重复上面的程序来获得二级目录的路径：vid_path
        for vid in os.listdir(pern_path):
            if vid == '1' or vid =='2' or vid =='HR_1':
                print(vid)
                vid_path = pern_path + '/' + vid
            
    		#现在我们获取了二级目录的路径：month_path
    		#再重复一次就可以得到三级目录的路径：data_path
                for img in os.listdir(vid_path):
                    #print(img)
                    img_path = vid_path + '/' + img
                    out=target_path+'/'+str(num)+'.jpg'
    			#到了这里，我们也已经获取了三级目录的路径：img_path
                    shutil.copy(img_path, out)
                    num+=1

def merge_img_of_false(data_path,target_path):
    if not os.path.exists(target_path):     #目标文件夹不存在就新建
        os.makedirs(target_path)

    num=1
    for pern in os.listdir(data_path):  #data_path路径下存在你的所有人的文件
        print(pern) #此时打印出来的是你data_path下的所有人
    	#这还没有进入一级目录，我们通过这一步仅仅获取了一级目录的文件名字
    	#那么我们把路径拼接起来，就可以得到第一级目录的路径名啦
        pern_path = data_path + '/' + pern 
    	#这里最重要的是别忘记加这个‘/’啦
    	#因为不加的话就不是一个完整的路径，你可以尝试print(data_path + year)看看效果
        #print(pern_path)#此时打印出来的就是一级目录的路径
    	#现在我们获取了一级目录：pern_path
    	#那么只用重复上面的程序来获得二级目录的路径：vid_path
        for vid in os.listdir(pern_path):
            if vid != '1' and vid !='2' and vid !='HR_1':
                print(vid)
                vid_path = pern_path + '/' + vid
            
    		#现在我们获取了二级目录的路径：month_path
    		#再重复一次就可以得到三级目录的路径：data_path
                for img in os.listdir(vid_path):
                    #print(img)
                    img_path = vid_path + '/' + img
                    out=target_path+'/'+str(num)+'.jpg'
    			#到了这里，我们也已经获取了三级目录的路径：img_path
                    shutil.copy(img_path, out)
                    num+=1

data_path1 = r'./video/train' #这里的data_path是文件的总路径
target_path1 = r'./video/train_true'   #目标文件夹
merge_img_of_true(data_path1, target_path1)
target_path2 = r'./video/train_false'   #目标文件夹
merge_img_of_false(data_path1, target_path2)
data_path3 = r'./video/test' #这里的data_path是文件的总路径
target_path3 = r'./video/test_true'   #目标文件夹
merge_img_of_true(data_path3, target_path3)
target_path4 = r'./video/test_false'   #目标文件夹
merge_img_of_false(data_path3, target_path4)