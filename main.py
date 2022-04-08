import cv2
import numpy as np
import os
from cProfile import label
import numpy as np
from numpy import *
import csv
import math

from sympy import true

###############################################
#分割目标区域
img=cv2.imread('D:/color/image/LAMP/Inked20220321154151_LI.jpg')
#print(img.max())
lower=np.array([0,43,46])
upper=np.array([34,255,255])
lower_red = np.array([156, 43, 46])
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

dst_name='D:/color/image/output/dst/01.jpg'   #保存
cv2.imwrite(dst_name,dst)

h1, w1 = dst.shape

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

#识别中心点
s_cen=0
for i in range(len(contours)):
    if len(contours[i])<10:   #去除小轮廓
        continue
    area = cv2.contourArea(contours[i])     #轮廓面积
    if area<w1*h1*0.1:        #去除小面积轮廓
        continue
    perimeter=cv2.arcLength(contours[i],True)  #轮廓周长
    sl=4*math.pi*area/(perimeter**2)
    if 0.8<sl<1.2:    #判断是否为圆形轮廓
        s_cir=area
        if s_cir>s_cen:
            s_cen=s_cir
            cen_i=i
C = cv2.moments(contours[cen_i]) # 计算轮廓的各阶矩,字典形式
#print(C)
cen_x = int(C["m10"] / C["m00"])
cen_y = int(C["m01"] / C["m00"])
cen_cir=[cen_x,cen_y]   #轮廓中心坐标

#分离目标区域
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
kernel = np.ones((5, 5), np.uint8)
dst_out = cv2.dilate(mask, kernel=kernel,iterations = 3)
dst_blue = cv2.erode(mask_blue, kernel=kernel,iterations = 2)
dst_name='D:/color/image/output/dst/1.jpg' 
cv2.imwrite(dst_name,dst_out)
dst_blue_name='D:/color/image/output/dst/blue.jpg' 
cv2.imwrite(dst_blue_name,dst_blue)
h, w = dst_out.shape
image = np.zeros([h, w], dtype=dst.dtype)

#提取轮廓信息
contours, cnt = cv2.findContours(dst_out, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )
contour_blue,_ = cv2.findContours(dst_blue, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )

cv2.circle(image, (cen_x-box[0][0],cen_y-box[0][1]), 5, 128, -1)#绘制中心点
cv2.putText(image,'center',(cen_x-box[0][0]-60, cen_y-box[0][1]+60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

#识别起始点附近的蓝色点
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
print(start_point)   
cv2.circle(image, (s_x,s_y), 5, 128, -1) 
cv2.putText(image,'start',(s_x-60,s_y-15),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2) 

#提取目标点
center=[]
n=1
for i in range(len(contours)):
    S=cv2.contourArea(contours[i])   #计算轮廓面积
    if S<w*h*0.00008 or S>w*h*0.0006:    
        continue
    C = cv2.moments(contours[i]) # 计算第一条轮廓的各阶矩,字典形式
    #print(C)
    center_x_ = int(C["m10"] / C["m00"])
    center_y_ = int(C["m01"] / C["m00"])
    xy=[center_x_,center_y_]
    l_c=point_distance(cen_cir,xy)
    #print(l_c,w,h)

    if l_c>w*0.37 or l_c>h*0.37 or l_c<w*0.25 or l_c<h*0.25:  #根据到中心点的距离，筛选目标点
        continue
    l_min=h1*w1
    for center_i in center:
        l_=point_distance(center_i,xy)
        l_min=min(l_,l_min)
    if l_min>w/15 or l_min>h/15:   #根据目标点之间的距离，筛选目标点
        center.append(xy)  #目标点的中心坐标(w,h)
        cv2.circle(out_img, (center_x_, center_y_), 2, 255, -1)#绘制中心点
        cv2.circle(image, (center_x_, center_y_), 5, 255, -1)#绘制中心点
        cv2.putText(image,str(n),(center_x_+30, center_y_+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        print(n,S,i)
        n+=1

cv2.drawContours(image, contours,-1, (255, 255, 255), 1)#绘制轮廓   
#cv2.imshow('contours',image)
#cv2.waitKey(0)
cv2.imwrite("D:/color/image/output/1.png", image)
#print(center)

#将目标点标签与实际标签一一对应
label=[]
cen=center.copy()
l_start=(h1**2 + w1**2)**0.5
for s in range(len(center)):  #找到起始点
    l1=point_distance(cen[s],start_point)
    if l1<l_start:
        sxy=cen[s]
        l_start=l1
label.append(sxy)
cen.remove(sxy)
for c in range(len(center)-1):
    l=(h1**2 + w1**2)**0.5
    for j in range(len(cen)):
        if label[c]==cen[j]:
            continue
        #L=(label[c][0]-cen[j][0])**2 + (label[c][1]-cen[j][1])**2
        L=point_distance(label[c],cen[j])
        if L<l:
            cxy=cen[j]
            l=L
    #print(l)
    label.append(cxy)
    cen.remove(cxy)
    
#print(label)
label.reverse()
#print(label)
for p in range(len(label)):
    cv2.putText(out_img,str(p+1),(label[p][0]+30, label[p][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    #cv2.putText(image,str(p+1),(label[p][0]+30, label[p][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
cv2.imwrite("D:/color/image/output/2.png", out_img)
#cv2.imwrite("D:/color/image/output/1.jpg", image)
    
#识别目标点颜色，并保存csv文件 
Label_=['细菌种类']
Color_=['颜色']
state_=['阴性/阳性']
lab=1
color_=[]
total_color_pixel=[]
for co in label:
    for y_ in range(-1,2):
        for x_ in range(int(-(1-y_**2)**0.5),int((1-y_**2)**0.5)+1):
            img_color=hsv[co[1]+y_,co[0]+x_,0]
            if img_color>100:
                img_color=img_color-180  #方便归一化
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

for i in range(len(color_)):
    color_[i]=color_[i]/(max_color-min_color)     #颜色标准化
    Label_.append('细菌'+str(i+1))
    Color_.append('%.2f'%color_[i])
    if color_[i]>(11-min_color)/(max_color-min_color):
        state_.append('阳性')
    else:
        state_.append('阴性')
    lab+=1

rows=zip(Label_,Color_,state_)
color_name='D:/color/image/output/细菌状态.csv'
with open(color_name,'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in rows:
        writer.writerow(row)
