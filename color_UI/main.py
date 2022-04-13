# coding=utf-8
import csv
import cv2
import math
# from numpy import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
#import pyqtgraph as pg
from scipy.optimize import curve_fit
import xlrd
import xlwt
from cProfile import label


from PyQt5.QtWidgets import QDialog,QApplication,QFileDialog,QGridLayout,QMainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon,QPainter, QBrush,QColor
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functools import partial

from color import Ui_MainWindow

import matplotlib as mpl

import pandas as pd
from pandas import DataFrame
# from openpyxl import load_workbook

class MyMainWindow(QMainWindow,Ui_MainWindow):

    def __init__(self):
        super(MyMainWindow,self).__init__()
        self.setupUi(self)
        self.setWindowTitle("至微生物  细菌感知检测系统")             #设置ui界面名字
        self.setWindowIcon(QIcon('2.ico'))                      #设置ui界面的图标，左上角
        self.setMinimumSize(0, 0)


        # ===========================================================
        #            对Ui界面外加的功能，如：设置Menu菜单等
        # ===========================================================
        self.statusBar().showMessage('准备就绪')        #软件底部的状态栏

        
        exitAct = QAction('退出(&E)', self)     #创建一个具有特定图标和“退出”标签的动作 exitAct = QAction(QIcon('exit.png'), '退出(&E)', self)
        exitAct.setShortcut('Ctrl+Q')           #定义快捷方式
        exitAct.setStatusTip('退出程序')        #当我们将鼠标指针悬停在菜单项上时，第三行创建状态栏显示在状态栏中
        #当我们选择这个特定的动作时，发出触发信号。 信号连接到QApplication小部件的quit()方法。 这终止了应用程序。
        exitAct.triggered.connect(qApp.quit)

        saveMenu = QMenu('保存格式(&S)', self)
        saveAct = QAction( 'xls', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('保存xls文件')
        saveasAct = QAction(QIcon('saveas.png'), 'csv(&O)', self)
        saveasAct.setStatusTip('保存csv文件')
        saveMenu.addAction(saveAct)
        saveMenu.addAction(saveasAct)
        saveAct.triggered.connect(self.save_file)
        saveasAct.triggered.connect(self.saveas_file)

        #newAct = QAction( '新建(&N)', self)
        #newAct.setShortcut('Ctrl+N')
        #newAct.setStatusTip('新建文件')
        
        #创建打开文件菜单，并连接到open_file函数和creat_table_show函数
        openfileAct = QAction('打开文件($O)', self)
        openfileAct.setShortcut('Ctrl+O')
        openfileAct.setStatusTip('打开文件')
        # 当我们选择这个特定的动作时，发出触发信号。 信号连接到open_file（）和creat_table_show方法。
        openfileAct.triggered.connect(self.open_file)
        #openfileAct.triggered.connect(self.creat_table_show)  # 点击打开文件按钮 并读取原数据显示


        menubar = self.menuBar()                #menuBar（）方法创建一个菜单栏
        fileMenu = menubar.addMenu('文件(&F)')  #增加了“&”这个符号，增加这个符号后，当我们按住“Alt+F”的时候就能快速打开文件这个菜单
        fileMenu.addAction(openfileAct)
        #fileMenu.addAction(newAct)              #addAction（）添加操作
        fileMenu.addMenu(saveMenu)              #addMenu（）创建文件菜单
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        helpMenu = menubar.addMenu('帮助(&H)')  #并行增加了一个帮助菜单


        # ===========================================================
        #       对Ui界面里面Qlabel或者Text Browser显示设置字体大小的
        # ===========================================================

        # 设置字体大小的
        font = QtGui.QFont()
        font.setFamily("Arial")  # 括号里可以设置成自己想要的其它字体
        font.setPointSize(18)  # 括号里的数字可以设置成自己想要的字体大小
        # self.Result.setFont(font)


        # ===========================================================
        #                       点击事件
        # ===========================================================
        # self.Open_File.clicked.connect(self.newSession)

        self.Open_File.clicked.connect(self.initiallize_parameters)  # 点击打开文件按钮 初始化参数
        self.Open_File.clicked.connect(self.open_file)  # 点击打开文件按钮 并显示文件路径
        # self.Open_File.clicked.connect(MyMainWindow.data_tianlong_matplotlib)
        #self.Open_File.clicked.connect(self.creat_table_show)  # 点击打开文件按钮 并读取原数据显示
        #self.Open_File.clicked.connect(self.Raw_data_judgment)
        #self.Open_File.clicked.connect(self.initiallize_parameters)  # 点击打开文件按钮 初始化参数



        self.start_run.clicked.connect(self.initiallize_tablet)  # 点击打开文件按钮 初始化表格
        #self.YuZhi.currentIndexChanged[int].connect(self.yuzhi_changed)

        #self.start_run.clicked.connect(self.data_tianlong_matplotlib)
        #self.start_run.clicked.connect(self.Data_Display_Cells_show)  # 第二页右下角计算数据的显示
        self.start_run.clicked.connect(partial(self.recognition))  # 点击按键开始运行


        self.Export.clicked.connect(self.save_file)    # 导出文件
    

    def initiallize_parameters(self):
        self.printDict = {}
        self.Label_ = []
        self.Color_ = []
        self.state_ = []
          

    #初始化表格
    def initiallize_tablet(self):
        for i in range(12):
            newItem = QTableWidgetItem('')
            newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.analysis_report_table.setItem(i, 2, newItem)
            self.analysis_report_table.setItem(i, 3, newItem)


    # ===========================================================
    #                       打开文件函数
    # ===========================================================

    def open_file(self):


        file_name1 = QFileDialog.getOpenFileName(self, '打开文件','./')   #打开文件关键函数

        self.file_name=file_name1[0]

        self.File_Path.setText(self.file_name)


    # ===========================================================
    #                       保存文件函数
    # ===========================================================
    def save_file(self):
        file_name2 = QFileDialog.getSaveFileName(self, '保存文件','./','Excel 工作簿(*.xls)')
        self.file_name2=file_name2[0]

        f = xlwt.Workbook('encoding = utf-8') #设置工作簿编码
        sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表
        rows=zip(self.Label_,self.Color_,self.state_)
        color_name=self.file_name2
        i = 0
        for row in rows:
            for j in range(len(row)):
                sheet1.write(i,j,row[j])
            i = i + 1    
        f.save(color_name) #保存文件


    def saveas_file(self):
        file_name2 = QFileDialog.getSaveFileName(self, '保存文件','./','CSV 文件(*.csv)')
        self.file_name2=file_name2[0]
        
        rows=zip(self.Label_,self.Color_,self.state_)
        color_name=self.file_name2
        with open(color_name,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)    


    # ===========================================================
    #                       最上角数据导出函数
    # ===========================================================
    def printExcel(self):
        data = []  # 存储全部数据
        data.append(self.printDict['label_'])
        data.append(self.printDict['color_'])
        data.append(self.printDict['state_'])

        df = pd.DataFrame(data=data, columns=None, index=None)
        #df.index.name = '自定义参数'

        # print(file_name)
        # writer = pd.ExcelWriter(file_name, engine='openpyxl')
        # book = load_workbook(file_name)
        # writer.book = book
        # writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        # data.to_excel(writer, sheet_name='汇总',index=False)
        # writer.save()
        # print('导出成功！！')

        import xlwings
        wb = xlwings.Book(self.file_name)
        last_sheet_name=wb.sheets[-1].name
        # 在wb中新建一张新的sheet.可以指定位置
        sht = wb.sheets.add(name="汇总", before=None, after=last_sheet_name)
        # df.values 不然会插入df的索引
        sht.range("A1").value = df
        wb.save()
        wb.close()
        print('导出成功！！')
        

    # ===========================================================
    #                       定义函数拟合函数
    # ===========================================================
    def func(x, A=None, B=None, C=None, Xc=None, y0=None):
        # return A / (1 + exp(-B * (x - Xc))) + y0
        # data=A * (x - Xc)
        # if data>=0:
        #     return B / (1 + exp(-A * (x - Xc)))+y0
        # else:
        #     return B*exp(A * (x - Xc)) / (1 + exp(A * (x - Xc)))+y0
        # return B / (C + exp(-A * (x - Xc))) + y0

        return B * (C + np.tanh(A * (x - Xc))) + y0

    # ===========================================================
    #                       定义画图的多种颜色函数
    # ===========================================================
    def muty_color(self):
        # 添加随机多种颜色选择
        color = []
        for name, hex in matplotlib.colors.cnames.items():
            color.append(name)
        return color

    # ===========================================================
    #                       识别
    # ===========================================================

    def recognition(self):
        ###############################################
        #分割目标
        #print(self.file_name)
        img=cv2.imread(self.file_name)
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

        ##################################################
        #定位读取
        h1, w1 = dst.shape

        #提取轮廓
        contours, cnt = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #contours中包含轮廓坐标信息
        
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
        
        h, w = dst_out.shape
        image = np.zeros([h, w], dtype=dst.dtype)

        #提取轮廓信息
        contours, cnt = cv2.findContours(dst_out, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )
        contour_blue,_ = cv2.findContours(dst_blue, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE )

        #识别起始点附近的蓝色点
        S_blue=0
        for i in range(len(contour_blue)):
            S_=cv2.contourArea(contour_blue[i])
            if S_>S_blue:
                S_blue=S_
                max_i=i
        C = cv2.moments(contour_blue[max_i]) # 计算第一条轮廓的各阶矩,字典形式
        #print(C)
        s_x = int(C["m10"] / C["m00"])
        s_y = int(C["m01"] / C["m00"])
        start_point=[s_x,s_y]
        #print(start_point)    

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
                #cv2.circle(out_img, (center_x_, center_y_), 2, 255, -1)#绘制中心点
                #cv2.circle(image, (center_x_, center_y_), 5, 255, -1)#绘制中心点
                #cv2.putText(image,str(n),(center_x_+30, center_y_+30),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                #print(n,S,i)
                n+=1

        #cv2.drawContours(image, contours,-1, (255, 255, 255), 1)#绘制轮廓   
        #cv2.imshow('contours',image)
        #cv2.waitKey(0)
        #cv2.imwrite("D:/color/image/output/1.png", image)
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
            label.append(cxy)
            cen.remove(cxy)
            
        #print(label)
        label.reverse()
        #print(label)
            
        #识别目标点颜色
        img_=cv2.imread(self.file_name)    #shape=(h,w,c),但label为(w,h)
        hsv_ = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV) 
        self.Label_=['细菌种类']
        self.Color_=['颜色']
        self.state_=['阴性/阳性']
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
        #print(min_color,max_color)
        #print(total_color_pixel)
        total_color_pixel=total_color_pixel-min_color
        #print(total_color_pixel)

        for co in range(0,len(total_color_pixel),5):
            color=(total_color_pixel[co]+total_color_pixel[co+1]+total_color_pixel[co+2]+total_color_pixel[co+3]+total_color_pixel[co+4])/5
            color_.append(color)
        
        for i in range(len(color_)):
            color_[i]=color_[i]/(max_color-min_color)     #颜色标准化
            self.Label_.append('细菌'+str(i+1))
            self.Color_.append('%.2f'%color_[i])
            if color_[i]>(11-min_color)/(max_color-min_color):
                self.state_.append('阳性')
            else:
                self.state_.append('阴性')
            lab+=1
        self.printDict = {'label_':self.Label_, 'color_':self.Color_, 'state_':self.state_}

        #写入UI表格
        for i in range(12):
            newItem1 = QTableWidgetItem(self.Color_[i+1])
            newItem1.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.analysis_report_table.setItem(i, 2, newItem1)
            newItem2 = QTableWidgetItem(self.state_[i+1])
            newItem2.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.analysis_report_table.setItem(i, 3, newItem2)

        
        '''
        for i in range(len(self.state_)):
            item = QtWidgets.QTableWidgetItem()
            _translate = QtCore.QCoreApplication.translate
            #MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            
            self.analysis_report_table.setItem(i, 2, item)
            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            #self.analysis_report_table.setItem(i, 3, item)
            #item = QtWidgets.QTableWidgetItem()
            #item.setTextAlignment(QtCore.Qt.AlignCenter)
            __sortingEnabled = self.analysis_report_table.isSortingEnabled()
            self.analysis_report_table.setSortingEnabled(False)
            item = self.analysis_report_table.item(i, 2)
            item.setText(self.state_[i+1])
            #item = self.analysis_report_table.item(i, 3)
            #item.setText(self.Color_[i+1])
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(18)
            item.setFont(font)
        '''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MyMainWindow()
    main.show()
    #app.installEventFilter(main)
    sys.exit(app.exec_())


#这个是一个打包的程序，在命令行输入即可，将程序打包成exe文件
#pyinstaller -F -i 33.ico main.py -w


#pyuic5 -o color.py color.ui    ui文件转py文件