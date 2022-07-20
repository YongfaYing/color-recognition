import cv2
import numpy as np
import os
from cProfile import label
import numpy as np
from numpy import *
import csv
import math
from sympy import true
import matplotlib.pyplot as plt

path='D:/color/image/20220708blindtest remove one blue/20220708blindtest remove one blue'    #文件夹地址
path_list = os.listdir(path)
name_list=[]
for file_name in path_list:
    if os.path.splitext(file_name)[1] == '.jpg':
        file_name=os.path.splitext(file_name)[0]   
        name_list.append(file_name)
name_list.sort()

tem=cv2.imread('D:/color/image/batch process/tem/4656_3496-high light.jpg')   #匹配样本文件地址

blue_tem=cv2.imread('D:/color/image/batch process/tem/blue_4656_3496.jpg')    #蓝色点样本文件地址

path_methods = path + '/methods_2'
if not os.path.exists(path_methods):
    os.mkdir(path_methods)
path_dir = path_methods +'/independent_norm'
if not os.path.exists(path_dir):
    os.mkdir(path_dir)    #一次只能创建一级目录

#####################################################################################
#分割目标区域
jpg = path + '/' + name_list[0] + '.jpg'
img=cv2.imread(jpg)
#print(img.max())
lower=np.array([0,26,46])
upper=np.array([40,255,255])
lower_red = np.array([140, 26, 46])
upper_red = np.array([180, 255, 255])
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

frame = img
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #颜色空间转换，将RGB图像转换成HSV图像

# get mask
mask1 = cv2.inRange(hsv, lower, upper)  #将hsv这个图像中低于lower_blue以及高于upper_blue的值，变成0；把处于lower_blue与upper_blue之间的值变成255
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask=mask1+mask2

#腐蚀膨胀
kernel = np.ones((5, 5), np.uint8)
dst = cv2.dilate(mask, kernel=kernel,iterations = 1)
'''
dst_dir=path+'/dst'
if not os.path.exists(dst_dir):   #判断输出路径是否存在
    os.mkdir(dst_dir)   #创建目录
dst_name=dst_dir+'/' + jpg_name + '.jpg'  #保存
cv2.imwrite(dst_name,dst)
'''
h1, w1 = dst.shape    #原图的高和宽

#提取轮廓信息
contours, cnt = cv2.findContours(dst, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )

#两点距离
def point_distance(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

#将轮廓信息重新排列
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

#识别芯片中心点
im=np.zeros([h1, w1], dtype=dst.dtype)
s_cen_pre=[]
#cv2.drawContours(im, contours,-1, (255, 255, 255), 1)#绘制轮廓 
for i in range(len(contours)):
    if len(contours[i])<10:   #去除小轮廓
        continue
    area = cv2.contourArea(contours[i])     #轮廓面积
    if area<w1*h1*0.01:        #去除小面积轮廓
        continue
    perimeter=cv2.arcLength(contours[i],True)  #轮廓周长
    sl=4*math.pi*area/(perimeter**2)

    x_contours,y_contours,w_contours,h_contours = cv2.boundingRect(contours[i])  #外接矩形
    al=area/(w_contours*h_contours)

    if 0.8<al*4/math.pi<1.2:    #判断是否为圆形轮廓
        s_cen_pre.append([i,sl,area])
if len(s_cen_pre)==1:
    cen_i = s_cen_pre[0][0]
else:
    s_cen=[]
    for sl_i in s_cen_pre:
        if w1*h1*0.08<sl_i[2]<w1*h1*0.3:
            s_cen.append(sl_i)
    if len(s_cen)==1:
        cen_i = s_cen[0][0]
    else:
        s_cen.sort(key=lambda x:x[2])
        cen_i = s_cen[-1][0]

    '''      
    C = cv2.moments(contours[i]) # 计算轮廓的各阶矩,字典形式
    #print(C)
    s_x = int(C["m10"] / C["m00"])
    s_y = int(C["m01"] / C["m00"])
    cv2.putText(im,str(i),(s_x, s_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    '''

#print(cen_i)
cv2.drawContours(im, contours[cen_i],-1, (255, 255, 255), 1)#绘制轮廓
'''
cen_dir=path+'/cen'
if not os.path.exists(cen_dir):   #判断输出路径是否存在
    os.mkdir(cen_dir)   #创建目录
cen_name=cen_dir + '/' + jpg_name + '.jpg'
cv2.imwrite(cen_name,im)
'''
C = cv2.moments(contours[cen_i]) # 计算轮廓的各阶矩,字典形式
#print(C)
cen_x = int(C["m10"] / C["m00"])
cen_y = int(C["m01"] / C["m00"])
cen_chip=[cen_x,cen_y]   #芯片中心坐标

#分离目标区域
'''
rect = cv2.minAreaRect(contours[cen_i])  #返回最小外接矩形
box = cv2.boxPoints(rect)          #获取矩形四个顶点
box = np.int0(box)                 #取整
box = box.reshape(4,2)
box = order_points(box)
box = np.int0(box) 
out_img=img[box[0][1]:box[2][1],box[0][0]:box[1][0],:]   #直接截取
cen_cir=[cen_x-box[0][0],cen_y-box[0][1]]
out_name='D:/color/image/output/dst/out_img.jpg' 
cv2.imwrite(out_name,out_img)
'''
area_cen = cv2.contourArea(contours[cen_i])
r = int((area_cen/math.pi)**0.5)+1
out_img = img[cen_y-r:cen_y+r, cen_x-r:cen_x+r]   #分割芯片区域
cen_cir = [r,r]
rate_ = r/45   #像素与芯片真实值的比值,但不是很精确

cut_dir=path_dir+'/cut'
if not os.path.exists(cut_dir):   #判断输出路径是否存在
    os.mkdir(cut_dir)   #创建目录
out_name=cut_dir + '/' + name_list[0] + '.jpg' 
cv2.imwrite(out_name,out_img)

###############################################################
#定位
frame = out_img
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #颜色空间转换，将RGB图像转换成HSV图像

# get mask
mask1 = cv2.inRange(hsv, lower, upper)  #将hsv这个图像中低于lower_blue以及高于upper_blue的值，变成0；把处于lower_blue与upper_blue之间的值变成255
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask=mask1+mask2
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

#腐蚀膨胀
#kernel = np.ones((5, 5), np.uint8)
#dst_out = cv2.dilate(mask, kernel=kernel,iterations = 3)
#dst_blue = cv2.erode(mask_blue, kernel=kernel,iterations = 2)
#dst_name='D:/color/image/output/dst/1.jpg' 
#cv2.imwrite(dst_name,dst_out)
'''
blue_dir=path+'/blue'
if not os.path.exists(blue_dir):   #判断输出路径是否存在
    os.mkdir(blue_dir)   #创建目录
dst_blue_name=blue_dir + '/' + jpg_name + '.jpg' 
cv2.imwrite(dst_blue_name,dst_blue)
'''
#h, w = dst_out.shape
h, w = 2*r, 2*r
image = np.zeros([h, w], dtype=dst.dtype)

#提取轮廓信息
#contours, cnt = cv2.findContours(dst_out, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )
#contour_blue,_ = cv2.findContours(dst_blue, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )

#cv2.circle(image, (cen_x-box[0][0],cen_y-box[0][1]), 5, 128, -1)#绘制中心点
#cv2.putText(image,'center',(cen_x-box[0][0]-60, cen_y-box[0][1]+60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
cv2.circle(image, (r,r), 5, 128, -1)#绘制中心点
cv2.putText(image,'center',(r-60, r+60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

#识别起始点附近的蓝色点
blue_res = cv2.matchTemplate(out_img,blue_tem,cv2.TM_CCORR_NORMED)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(blue_res)   #得到匹配度最高的坐标
start_point = [max_loc[0]+blue_tem.shape[1]//2, max_loc[1]+blue_tem.shape[0]//2]    #蓝色点的位置

#blue_img = np.zeros([h, w], dtype=dst.dtype)
cv2.rectangle(out_img,(max_loc[0], max_loc[1]),[max_loc[0]+blue_tem.shape[1], max_loc[1]+blue_tem.shape[0]],(255,255,0),2)
'''
S_blue=0
for i in range(len(contour_blue)):
    S_=cv2.contourArea(contour_blue[i])
    if S_>S_blue:
        S_blue=S_
        max_i=i
C = cv2.moments(contour_blue[max_i]) # 计算轮廓的各阶矩,字典形式
#print(C)
s_x = int(C["m10"] / C["m00"])
s_y = int(C["m01"] / C["m00"])
start_point=[s_x,s_y]
#print(start_point)   
cv2.circle(image, (s_x,s_y), 5, 128, -1) 
cv2.putText(image,'start',(s_x-60,s_y-15),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2) 
'''
#提取目标点
center=[]
n=1
#匹配
res = cv2.matchTemplate(out_img,tem,cv2.TM_CCORR_NORMED)

#相似阈值
threshold = 1    

while len(center)<8:
    threshold-=0.005
    loc = np.where(res>=threshold)
    
    for pt in zip(*loc[::-1]):#*号表示可选参数
        bottom_right = [pt[0]+tem.shape[1],pt[1]+tem.shape[0]]
        b=True
        for i in range(len(center)):
            if point_distance(bottom_right,center[i])<tem.shape[0]*2:
                b=False
                break
        if b:
            if 27*rate_ < point_distance(bottom_right,cen_cir) < 37*rate_:
                center.append([pt[0]+tem.shape[1]//2,pt[1]+tem.shape[0]//2])
                #cv2.rectangle(out_img,pt,bottom_right,(0,255,0),2)
print(threshold)


#将目标点标签与实际标签一一对应
label_pre=[]
cen=center.copy()
l_start=(h1**2 + w1**2)**0.5
for s in range(len(center)):  #找到起始点
    l1=point_distance(cen[s],start_point)
    if l1<l_start:
        sxy=cen[s]
        l_start=l1
label_pre.append(sxy)
cen.remove(sxy)
#二次筛选：由于1号孔距离过近，去掉1号孔之后，再根据中心距筛选一次
if len(center)>8:
    center=[label_pre[0]]
    cv2.rectangle(out_img,[center[0][0]-tem.shape[1]//2,center[0][1]-tem.shape[0]//2],[center[0][0]+tem.shape[1]//2,center[0][1]+tem.shape[0]//2],(0,255,0),2)
    for s in cen:
        if 30*rate_ < point_distance(s,cen_cir) < 37*rate_:
            center.append(s)
            #cv2.rectangle(out_img,[s[0]-tem.shape[1]//2,s[1]-tem.shape[0]//2],[s[0]+tem.shape[1]//2,s[1]+tem.shape[0]//2],(0,255,0),2)
else:
    for s in center:
        cv2.rectangle(out_img,[s[0]-tem.shape[1]//2,s[1]-tem.shape[0]//2],[s[0]+tem.shape[1]//2,s[1]+tem.shape[0]//2],(0,255,0),2)
    
match_dir=path_dir+'/match'
if not os.path.exists(match_dir):   #判断输出路径是否存在
    os.mkdir(match_dir)   #创建目录
match_name=match_dir + '/' + name_list[0] + '.jpg' 
cv2.imwrite(match_name,out_img)


cen=center.copy()
cen.remove(label_pre[0])
for c in range(len(center)-1):
    l=(h1**2 + w1**2)**0.5   #初始l,斜角边长度
    for j in range(len(cen)):
        if label_pre[c]==cen[j]:
            continue
        #L=(label[c][0]-cen[j][0])**2 + (label[c][1]-cen[j][1])**2
        L=point_distance(label_pre[c],cen[j])
        if L<l and rate_*17 < L < rate_*22:
            cxy=cen[j]
            l=L
    #print(l)
    label_pre.append(cxy)
    cen.remove(cxy)
#三次筛选
label=[]
if len(label_pre)>8:
    base_l = point_distance(cen_cir,label_pre[0])
    #print('base_l:',base_l)
    for la in label_pre:
        la_l = point_distance(cen_cir,la)
        #print('la_l:',la_l)
        if base_l-w*0.05 < la_l < base_l+w*0.1:
            label.append(la)
else:
    label = label_pre

if len(label)>8:            
    label=label[0:8]   

##############################################################################################
#根据得到的反应孔坐标去识别颜色，一批照片采用同一组坐标
xy_csv = []
state_csv = []
num = 0
for jpg_name in name_list:
    print(num)
    xy_csv.append([jpg_name[8:12]])
    state_csv.append([''])

    jpg = path + '/' + jpg_name + '.jpg'
    img=cv2.imread(jpg)   
    out_img = img[cen_y-r:cen_y+r, cen_x-r:cen_x+r]
    frame = out_img
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #颜色空间转换，将RGB图像转换成HSV图像

    #label可视化
    for p in range(len(label)):
        cv2.circle(out_img, (label[p][0], label[p][1]), 2, 255, 1)
        cv2.putText(out_img,str(p+1),(label[p][0]+30, label[p][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
        #cv2.putText(image,str(p+1),(label[p][0]+30, label[p][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    cv2.circle(out_img, (r,r), 5, 128, 3)
    cv2.putText(out_img,'center',(r+30, r+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
    label_dir=path_dir+'/label'
    if not os.path.exists(label_dir):   #判断输出路径是否存在
        os.mkdir(label_dir)   #创建目录
    label_name=label_dir + '/' + jpg_name + '.jpg' 
    cv2.imwrite(label_name , out_img)
    #cv2.imwrite("D:/color/image/output/1.jpg", image)
        
    #识别目标点颜色，并保存csv文件 
    Label_=['细菌种类']
    Color_=['颜色']
    state_=['阴性/阳性']
    lab=1
    color_=[]
    total_color_pixel=[]
    #img_colo=hsv[217,278]
    #print('b: ',img_colo)
    
    #独立归一化标准
    #查找需要的点
    
    #点的搜索算法
    def search_point(co,point_img,point_img_copy,point_img_size):
        from collections import deque
        #四个方向
        dirs = [lambda y,x:(y,x+1),    #右边
            lambda y,x:(y+1,x),    #下边
            lambda y,x:(y,x-1),     #左边
            lambda y,x:(y-1,x)    #上边
            ] 
        x,y = co[0],co[1]
        current_queue = deque()  #创建新队列
        current_queue.append([x,y])
        searched_point = []   #存放所有出队的数
        color_point = []   #所有点的像素值

        while current_queue:
            current_point = current_queue.popleft()   #(x,y)
            point_img_copy[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0] = -1   #标记已经走过的位置,等于-1后会变成255   #截取区域要与原坐标有个对应
            searched_point.append(current_point)
            if point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0]>100:
                color_current = point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0]
                color_current -= 180
            else:
                color_current = point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0]
            color_point.append(color_current)  #当前点像素值

            #print(current_point, color_current)
            for dir in dirs:
                next_point = dir(current_point[0],current_point[1])
                if point_img_copy[next_point[1]-co[1]+point_img_size][next_point[0]-co[0]+point_img_size][0] == 255:  #已经走过的位置
                    continue
                if point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0]>100:
                    color_next = point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0] - 180
                else:
                    color_next = point_img[current_point[1]-co[1]+point_img_size][current_point[0]-co[0]+point_img_size][0]
                if abs(color_next-mean(color_point))<2:  #颜色差别不大的点才能加入
                    current_queue.append(next_point)
                    point_img_copy[next_point[1]-co[1]+point_img_size][next_point[0]-co[0]+point_img_size][0] = -1   #标记已经走过的位置   #截取区域要与原坐标有个对应
        return color_point,searched_point,point_img_copy  #searched_point用于可视化


    i=0
    point_img_size = 35  #搜索的区域大小，此值为边长的一半
    for co in label:
        #截取目标点附近一片区域
        point_img = hsv[co[1]-point_img_size:co[1]+point_img_size,co[0]-point_img_size:co[0]+point_img_size]   #29后面要适应性调整   #co[0]和co[1]可能要反过来
        #print(point_img)
        point_img_copy = point_img.copy()
        point_img_copy_rgb = cv2.cvtColor(point_img_copy, cv2.COLOR_HSV2BGR)
        vis_dir=path_dir+'/vis'
        if not os.path.exists(vis_dir):   #判断输出路径是否存在
            os.mkdir(vis_dir)   #创建目录
        vis_dir_name = vis_dir + '/' + jpg_name
        if not os.path.exists(vis_dir_name):   #判断输出路径是否存在
            os.mkdir(vis_dir_name)
        vis_name=vis_dir_name + '/' + str(i) + '.jpg'
        cv2.imwrite(vis_name , point_img_copy_rgb)

        vis_point_dir = path_dir+'/vis_point'
        if not os.path.exists(vis_point_dir):   #判断输出路径是否存在
            os.mkdir(vis_point_dir)
        vis_point_index = vis_point_dir + '/' + str(i+1)
        if not os.path.exists(vis_point_index):   #判断输出路径是否存在
            os.mkdir(vis_point_index)
        vis_point_name = vis_point_index + '/' + jpg_name + '.jpg'
        cv2.imwrite(vis_point_name , point_img_copy_rgb)

        #cv2.rectangle(point_img_vis,[co[0]-point_img_size,co[1]+point_img_size],[co[0]+point_img_size,co[1]-point_img_size],(0,255,0),2)   #可视化，看看截取的部分对不对
        i+=1
        
        color_point, searched_point, point_img_copy = search_point(co,point_img=point_img,point_img_copy=point_img_copy,point_img_size=point_img_size)
        total_color_pixel.append(color_point)
        point_img_copy = cv2.cvtColor(point_img_copy, cv2.COLOR_HSV2BGR)

        #print(len(color_point))
        #print(mean(color_point))
        #可视化
        #for p in searched_point:
        #    cv2.circle(point_img_vis, (p[0],p[1]), 1, (255,0,255), -1)
        
        vis_name_point=vis_dir_name + '/diff_2_' + str(i) + '.jpg' 
        cv2.imwrite(vis_name_point , point_img_copy)

        vis_point_name_diff = vis_point_index + '/diff_2_' + jpg_name + '.jpg'
        cv2.imwrite(vis_point_name_diff, point_img_copy)
    '''
    total = [i for item in total_color_pixel for i in item]
    total.sort()
    min_color=min(total)
    max_color=max(total)
    #max_color=max(max_color)
    total_color_pixel=total_color_pixel-min_color
    '''
    for i in range(len(total_color_pixel)):
        color = mean(total_color_pixel[i])
        color_.append(color)
    print(color_)

    #min_color=min(color_)
    #max_color=max(color_)
    #color_ = color_ - min_color

    '''
    for co in label:
        for y_ in range(-1,2):
            for x_ in range(int(-(1-y_**2)**0.5),int((1-y_**2)**0.5)+1):
                img_color=hsv[co[1]+y_,co[0]+x_,0]
                if img_color>100:
                    img_color=img_color-180  
                total_color_pixel.append(img_color)
             
    min_color=min(total_color_pixel)
    max_color=max(total_color_pixel)
    print(min_color,max_color)
    print(total_color_pixel)
    total_color_pixel=total_color_pixel-min_color
    print(total_color_pixel)
    
    for co in range(0,len(total_color_pixel),5):
        color=(total_color_pixel[co]+total_color_pixel[co+1]+total_color_pixel[co+2]+total_color_pixel[co+3]+total_color_pixel[co+4])/5
        color_.append(color)
    print(color_)
    '''
    '''
    for i in range(len(color_)):
        color_[i]=color_[i]/(max_color-min_color)    #颜色标准化
        Label_.append('细菌'+str(i+1))
        Color_.append('%.2f'%color_[i])
        if color_[i]>(11-min_color)/(max_color-min_color):
            state_.append('阳性')
        else:
            state_.append('阴性')
        lab+=1
        xy_csv[num].append(float('%.2f'%color_[i]))
    '''
    #不归一化
    for i in range(len(color_)):
        Label_.append('细菌'+str(i+1))
        Color_.append('%.2f'%color_[i])
        if color_[i]>11:
            state_.append('阳性')
            state_csv[num].append('阳性')
        else:
            state_.append('阴性')
            state_csv[num].append('阴性')
        lab+=1
        xy_csv[num].append(float('%.2f'%color_[i]))
        

    '''
    #去除环境变化
    base = color_[0]
    for i in range(len(color_)):
        color_[i]=(color_[i]-base)/(max_color-min_color-base)    #颜色标准化
        color_[i]-=color_[0]
        if color_[i]<0:
            color_[i]=0
        Label_.append('细菌'+str(i+1))
        Color_.append('%.2f'%color_[i])
        if color_[i]>(11-min_color-base)/(max_color-min_color-base):
            state_.append('阳性')
        else:
            state_.append('阴性')
        lab+=1
        xy_csv[num].append(float('%.2f'%color_[i]))
    '''
    num+=1

#绘制折线图
xy_csv=list(map(list,zip(*xy_csv)))  #转置
print(xy_csv)

plt.figure(dpi=128, figsize=(8,6))  #绘图窗口大小
plt.title('细菌反应孔变化情况')
plt.rcParams['font.sans-serif'] = ['SimHei']  #显示汉字
plt.rcParams['axes.unicode_minus']=False      #解决负号不显示问题
plt.xlabel('反应孔')
plt.ylabel('颜色变化')
x=[0,2]
for i in range(len(xy_csv[0])-2):
    x.append(x[-1]+3)
for i in range(1,len(xy_csv)):
    plt.plot(x, xy_csv[i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小


plt.legend(['1','2','3','4','5','6','7','8'], loc=2, ncol=1)  #设置折线名称
#plt.show()
output_dir = path_dir + '/变化图'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_name = output_dir + '/细菌各反应孔变化.png'
plt.savefig(output_name,bbox_inches='tight')

for png_i in range(1,9):
    plt.figure(dpi=128, figsize=(8,6))  #绘图窗口大小
    plt.title('细菌反应孔变化情况-'+str(png_i)+'号孔')
    plt.rcParams['font.sans-serif'] = ['SimHei']  #显示汉字
    plt.xlabel('时间')
    plt.ylabel('颜色变化')
    x=[0,2]
    for i in range(len(xy_csv[0])-2):
        x.append(x[-1]+2)
    plt.plot(x, xy_csv[png_i], marker='o', markersize=3)
    output_name_i = output_dir + '/' + str(png_i) + '号孔变化'
    plt.savefig(output_name_i,bbox_inches='tight')

#rows=zip(xy_csv[0],)
state_csv=list(map(list,zip(*state_csv)))  #转置

color_name=path_dir+'/细菌状态.csv'
with open(color_name,'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in xy_csv:
        writer.writerow(row)
with open(color_name,'a',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in state_csv:
        writer.writerow(row)