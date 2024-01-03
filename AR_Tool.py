from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from objloader import *
import cv2.aruco as aruco
import yaml
import imutils
import time
import sys
import glob

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from MatrixTransform import extrinsic2ModelView, intrinsic2Project 
from Filter import Filter
from inter_ui import *

counter = 0
angles=[0,0,0,0,0,0] #base shoulder elbow 

class OpenGLGlyphs(QMainWindow):

    def __init__(self):

        QMainWindow.__init__(self)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QIcon("./Logos/LogoAR.png"))
        self.ui.btn_page_1.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.btn_page_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_1))
        self.ui.btn_page_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.camara()
        self.progress()
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.main)      
        self.ui.Bnext.clicked.connect(self.nextdef)          
        self.ui.Batras.clicked.connect(self.atrasdef)
        self.ui.btn_close.clicked.connect(self.cerrar)
        self.ui.btn_minimize.clicked.connect(self.minimize)   
        self.ui.labSlider.valueChanged.connect(self.scalslid)
        self.ui.labSlider2.valueChanged.connect(self.posslid)
        self.ui.radioButton.toggled.connect(self.camara)
        self.ui.rotar.clicked.connect(self.Brotacion)
        self.ui.tableWidget.clicked.connect(self.lista)
        self.ui.BResetAnim.clicked.connect(self.resetanim) 
        self.ui.ButtOpac.setCheckable(True)
        self.ui.ButtOpac.clicked.connect(self.Opacity) 
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.controlTimer()

        #-----------------------------------------------------------------------------
        # initialise webcam and start thread
        self.image_w, self.image_h = map(int, (self.webcam.get(3), self.webcam.get(4)))
        self.model_scale=0.03
        self.model_posx=0
        self.model_posy=0
        self.model_posz=0
        #SR
        self.cam_matrix,self.dist_coefs,rvecs,tvecs = self.get_cam_matrix("data3.yaml")
        #
        self.projectMatrix = intrinsic2Project(self.cam_matrix, self.image_w, self.image_h, 0.01, 100.0)
        self.pre_extrinsicMatrix = None
        self.filter = Filter()
        self.model = None
        self.File = None
        self.texid = None
        #SR Rotacion de modelo general
        self.rotate=0
        self.starttime = time.time()
        self.pausetime = 0
        self.resumetime = 0
        self.angle=0
        #tiempomejora deteccion
        self.tiemposi=0
        self.tiempono=0
        self.tiempodif=0
        #
        #Seleccion de modelos
        self.selectmodel=0
        self.animate=1     

        self.ui.tableWidget.setCurrentCell(self.selectmodel,0)

        def moveWindow(e):
            # Detect if the window is  normal size
            # ###############################################  
            if self.isMaximized() == False: #Not maximized
                # Move window only when window is normal size  
                # ###############################################
                #if left mouse button is clicked (Only accept left mouse button clicks)
                if e.buttons() == Qt.LeftButton:  
                    #Move window 
                    self.move(self.pos() + e.globalPos() - self.clickPosition)
                    self.clickPosition = e.globalPos()
                    e.accept()  

        self.ui.frametop.mouseMoveEvent = moveWindow
        QSizeGrip(self.ui.sizegrip)

    def mousePressEvent(self, event):
        # ###############################################
        #Get the current position of the mouse
        self.clickPosition = event.globalPos()

    def get_cam_matrix(self,file): #obtener las variables del archivo yaml generado y pasarlas a constantes
        with open(file) as f:
            loadeddict = yaml.load(f)
            cam_matrix = np.array(loadeddict.get('camera_matrix'))
            dist_coeff = np.array(loadeddict.get('dist_coeff'))
            rvecs = np.array(loadeddict.get('rvecs'))
            tvecs = np.array(loadeddict.get('tvecs'))
            return cam_matrix,dist_coeff,rvecs,tvecs

 
    def initOpengl(self, width, height): #inicializacion de opengl con variables por predeterminadas

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(37, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        #Disable Opacity
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)

        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texid = glGenTextures(1)

        global counter
         # Load 3d object

        for var in range(1, 50):#nmodelos +1
                setattr(self,'File%d'%var,'./models/Parts/'+str(var)+'.obj')
                charger=getattr(self,'File%d'%var)
                setattr(self,'model%d'%var,OBJ(charger,swapyz=True))
                counter +=101/49
                self.progress()
        
        self.Sel_Model()
        self.control_animacion()
 
    def draw_scene(self): #superposicion del background con los modelos 3D
        # get image from webcam
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glLoadIdentity()
 
        # get image from webcam
        ret,image = self.webcam.read()
        # image = imutils.resize(image,width=640)
 
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
         
        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texid)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self.draw_background()
        glPopMatrix()
 
        # handle glyphs
        image = self.draw_objects(image)
        glutSwapBuffers()

    def draw_background(self): #dibujo del background (camara)
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd()
 
    def draw_objects(self, image): #dibujo de los modelos 3d sobre el marcador aruco
        """[draw models with opengl]
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.07})
        """
        # aruco data
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)      
        parameters =  aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 1

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        
        rvecs, tvecs, model_matrix = None, None, None
        
        if ids is not None and corners is not None:
            rvecs, tvecs, _= aruco.estimatePoseSingleMarkers(corners, 0.05 , self.cam_matrix, self.dist_coefs)
 
        projectMatrix = intrinsic2Project(self.cam_matrix, width, height, 0.01, 100.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(projectMatrix)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if tvecs is not None:
            self.tiemposi=time.time()
            if self.filter.update(tvecs): # the mark is moving
                model_matrix = extrinsic2ModelView(rvecs, tvecs)
            else:
                model_matrix = self.pre_extrinsicMatrix
        else:
            self.tiempono=time.time()
            self.tiempodif=self.tiempono-self.tiemposi
            if self.tiempodif <= 0.15:
                model_matrix = self.pre_extrinsicMatrix

        
        if model_matrix is not None:     
            self.pre_extrinsicMatrix = model_matrix
            glLoadMatrixf(model_matrix)
            glScaled(self.model_scale, self.model_scale, self.model_scale)
            
            #SR
            glTranslatef(0,self.model_posy,0)  
            glRotatef(self.angle, 0, 0, 1) #rotacion de modelo 3d sobre el marcador en el eje z  
                   

           
            
            if self.selectmodel==0:
                self.animate = 1
                self.BrazoCompleto()
            elif self.selectmodel==1:
                self.animate = 0
                glCallList(self.model.gl_list)
            elif self.selectmodel==2:
                self.animate = 1
                glPushMatrix()
                glRotatef(-90, 1, 0, 0)
                glTranslatef(-0.001,-0.071,-0.003)
                self.Axis1()
                glPopMatrix()
                glPushMatrix()
                glRotatef(angles[0], 0, 0, 1);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==3:
                self.animate = 0
                glCallList(self.model.gl_list)
            elif self.selectmodel==4:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis1()
            elif 5 <= self.selectmodel <=8:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)

            elif self.selectmodel==9:
                self.animate = 1
                glPushMatrix()
                glRotatef(90, 0, 0, 1)
                glTranslatef(0.004,-0.188,-0.001)
                self.Axis2()
                glPopMatrix()
                glPushMatrix()
                glRotatef(angles[1], 1, 0, 0);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==10:
                self.animate = 0
                glCallList(self.model.gl_list)
            elif self.selectmodel ==11:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis2()
            elif 12 <= self.selectmodel <=20:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)

            elif self.selectmodel==21:
                self.animate = 1
                glScalef(2,2,2)
                glPushMatrix()
                glRotatef(90, 0, 0, 1)
                glTranslatef(0,-0.342,0)
                glRotatef(-angles[2], 0, 1, 0);
                glCallList(self.model3.gl_list)        
                glTranslatef(0,0.482,0)
                self.Axis3()
                glPopMatrix()
                glPushMatrix()
                glRotatef(angles[2], 1, 0, 0);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==22:
                self.animate = 0
                glScalef(2,2,2)
                glCallList(self.model.gl_list)
            elif self.selectmodel==23:
                self.animate = 1
                glScalef(3,3,3)
                glPushMatrix()
                glTranslatef(0,-0.482,0)
                glCallList(self.model3.gl_list)
                glPopMatrix()
                self.Axis3()
            elif 24 <= self.selectmodel <=33:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)

            elif self.selectmodel==34:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis4()
                glPushMatrix()
                glTranslatef(0,0.121,0)
                glCallList(self.model25.gl_list)
                glRotatef(angles[3], 0, 1, 0);
                glCallList(self.model26.gl_list)
                glPopMatrix()
                glPushMatrix()
                glTranslatef(-0.003,0.503,0)
                glRotatef(angles[3], 0, 1, 0);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==35:
                self.animate = 0
                glScalef(3,3,3)
                glCallList(self.model.gl_list)
            elif self.selectmodel==36:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis4()
            elif 37 <= self.selectmodel <=43:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)
                
            elif self.selectmodel==44:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis5()
                glPushMatrix()
                glTranslatef(0,-0.234,0)      
                glCallList(self.model25.gl_list)
                glPopMatrix()
                glPushMatrix()
                glTranslatef(-0.038,0.312,0)
                glRotatef(angles[4], 1, 0, 0);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==45:
                self.animate = 0
                glScalef(3,3,3)
                glCallList(self.model.gl_list)
            elif self.selectmodel==46:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis5()
            elif 47 <= self.selectmodel <=59:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)

            elif self.selectmodel==60:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis6_1()
                self.Axis6_2()
                glPushMatrix()
                glTranslatef(0,-0.187,0)      
                glCallList(self.model25.gl_list)
                glPopMatrix()
                glPushMatrix()
                glTranslatef(-0.035,0.663,0)
                glRotatef(angles[5], 0, 1, 0);
                glCallList(self.model.gl_list)
                glPopMatrix()
            elif self.selectmodel==61:
                self.animate = 0
                glScalef(3,3,3)
                glCallList(self.model.gl_list)
            elif self.selectmodel==62:
                self.animate = 1
                glScalef(3,3,3)
                self.Axis6_1()
                self.Axis6_2()
            elif 63 <= self.selectmodel <=78:
                glScalef(3,3,3)
                self.animate = 0
                glCallList(self.model.gl_list)

            elif self.selectmodel==79:
                self.animate = 1
                glScalef(3,3,3)
                glPushMatrix()
                glTranslatef(0,-0.234,0)      
                glCallList(self.model25.gl_list)
                glPopMatrix()
                self.Axis5()
                glTranslatef(0,-0.047,0)
                self.Axis6_1()
                glPushMatrix()
                glTranslatef(0,0.36,0)
                glRotatef(angles[4],1,0,0)
                glTranslatef(0,-0.36,0)
                self.Axis6_2()
                glPopMatrix()


            
        self.rotation()
        self.iconanimate()
        

    def BrazoCompleto(self):
        glCallList(self.model1.gl_list)
        glTranslatef(-0.005,0.02,0.446)
        glRotatef(angles[0], 0, 0, 1);
        glCallList(self.model2.gl_list)
        glPushMatrix()
        glTranslatef(0,0,0.4)
        glRotatef(-90, 1, 0, 0);
        glCallList(self.model3.gl_list)
        glTranslatef(0.047,-0.181,0.626)
        glRotatef(90, 1, 0, 0);
        glRotatef(90, 0, 0, 1);
        glCallList(self.model3.gl_list)
        glPopMatrix()

        glTranslatef(-0.177,0.611,0.602)
        glRotatef(angles[1], 1, 0, 0);
        glCallList(self.model7.gl_list)

        glTranslatef(0.31,-0.03,1.71)
        glRotatef(angles[2], 1, 0, 0);
        glCallList(self.model16.gl_list)
        glPushMatrix()
        glTranslatef(0.35,0,0)
        glRotatef(90,0,0,1)
        glCallList(self.model3.gl_list)
        glScaled(0.8,0.8,0.8)
        glTranslatef(-0.65,0.465,0.365)
        glRotatef(-90,0,0,1)      
        glCallList(self.model3.gl_list)
        glTranslatef(0,0,-0.28)        
        glCallList(self.model3.gl_list)
        glTranslatef(0,0,-0.28)        
        glCallList(self.model3.gl_list)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0,0.597,0.083)  
        glCallList(self.model25.gl_list) 
        glCallList(self.model26.gl_list)
        glPopMatrix()

        glTranslatef(0,1.01,0.08)
        glRotatef(angles[3], 0, 1, 0);
        glCallList(self.model24.gl_list)

        glTranslatef(-0.04,0.17,0)
        glRotatef(angles[4], 1, 0, 0);
        glCallList(self.model31.gl_list)

        glTranslatef(0,0.3,0)
        glRotatef(angles[5], 0, 1, 0);
        glCallList(self.model43.gl_list)

    def Axis1(self):
        glPushMatrix();
        glTranslatef(0,-0.314,0)
        glCallList(self.model3.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(angles[0]*5.9375, 0, 1, 0);
        glCallList(self.model4.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.124,0,0)
        glRotatef(-angles[0]*2.375-1, 0, 1, 0);#3.61
        glCallList(self.model5.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.073,0,0.101)
        glRotatef(-angles[0]*2.375+2, 0, 1, 0);
        glCallList(self.model5.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.057,0,-0.11)
        glRotatef(-angles[0]*2.375+1, 0, 1, 0);
        glCallList(self.model5.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0,0)
        glRotatef(-angles[0], 0, 1, 0);
        glCallList(self.model6.gl_list)
        glPopMatrix();

    def Axis2 (self):
        glPushMatrix();
        glTranslatef(0,-0.277,0)
        glCallList(self.model3.gl_list)
        glPopMatrix();
        glPushMatrix();
        glCallList(self.model8.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(-angles[1]*93.6, 0, 1, 0);
        glCallList(self.model9.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.073,-0.063)
        glRotatef(angles[1]*156, 0, 1, 0);
        glCallList(self.model10.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.156,-0.063)
        glRotatef(angles[1]*156, 0, 1, 0);
        glCallList(self.model11.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.19,-0.111)
        glRotatef(-angles[1]*156, 0, 0, 1);
        glCallList(self.model12.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.19,0)
        glRotatef(-angles[1]*156, 0, 0, 1);
        glCallList(self.model13.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.089,0.171,0.067)
        glRotatef(-angles[1]*3+2, 0, 1, 0);
        glCallList(self.model14.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.153,0)
        glRotatef(-angles[1], 0, 1, 0);
        glCallList(self.model15.gl_list)
        glPopMatrix();

    def Axis3 (self):
        glPushMatrix();
        glCallList(self.model17.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(angles[2]*213.33333333, 0, 1, 0);
        glCallList(self.model18.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.043,0.084,-0.026)
        glRotatef(-angles[2]*120, 0, 1, 0);
        glCallList(self.model19.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.044,0.144,-0.024)
        glRotatef(-angles[2]*120, 0, 1, 0);
        glCallList(self.model11.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(-60.28, 0, 1, 0);
        glTranslatef(0,0.182,-0.096)
        glRotatef(angles[2]*120, 0, 0, 1);
        glCallList(self.model12.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(-60.28, 0, 1, 0);
        glTranslatef(0,0.182,0.047)
        glRotatef(angles[2]*120, 0, 0, 1);
        glCallList(self.model20.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.079,0.153,-0.03)
        glRotatef(angles[2]*2.85714285714, 0, 1, 0);
        glCallList(self.model21.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.0125,0.126,0.083)
        glRotatef(-angles[2]*2.06896551724, 0, 1, 0);
        glCallList(self.model22.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.121,0)
        glRotatef(-angles[2]-1.3, 0, 1, 0);
        glCallList(self.model23.gl_list)
        glPopMatrix();

    def Axis4 (self):
        glPushMatrix();
        glScalef(0.8,0.8,0.8)
        glTranslatef(0,-1.247,0)
        glCallList(self.model3.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(-angles[3]*25.3968253968, 0, 1, 0);
        glTranslatef(0,-0.537,0)
        glCallList(self.model27.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(-angles[3]*25.3968253968, 0, 1, 0);
        glCallList(self.model28.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.064,0.081,0)
        glRotatef(angles[3]*5.71428571429, 0, 1, 0);        
        glCallList(self.model29.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.064,0.081,0)
        glRotatef(angles[3]*5.71428571429, 0, 1, 0);        
        glCallList(self.model29.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.125,0)
        glRotatef(angles[3], 0, 1, 0);        
        glCallList(self.model30.gl_list)
        glPopMatrix();

    def Axis5 (self):
        glPushMatrix();
        glScalef(0.8,0.8,0.8)
        glTranslatef(0,-1.68,-0.284)
        glCallList(self.model3.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-1.075,-0.227)
        glRotatef(-angles[4]*1.68, 0, 1, 0);        
        glCallList(self.model32.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.761,-0.145)
        glRotatef(16.22, 1, 0, 0);
        glRotatef(-angles[4]*1.68, 0, 1, 0);        
        glCallList(self.model33.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.431,-0.071)
        glRotatef(-angles[4]*1.68, 0, 1, 0);        
        glCallList(self.model34.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.298,-0.071)
        glRotatef(-angles[4]*1.68, 0, 1, 0);        
        glCallList(self.model35.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.297,0)
        glRotatef(angles[4], 0, 1, 0);        
        glCallList(self.model36.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.122,0)
        glRotatef(angles[4], 0, 1, 0);        
        glCallList(self.model37.gl_list)
        glPopMatrix();
        glPushMatrix();
        glRotatef(angles[4], 0, 1, 0);        
        glCallList(self.model38.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.132,0.092,-0.002)       
        glCallList(self.model41.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.184,0.312,0)
        glRotatef(angles[4], 1, 0, 0);        
        glCallList(self.model39.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.178,0.019,-0.001)     
        glCallList(self.model40.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0.312,0)    
        glRotatef(angles[4], 1, 0, 0);  
        glCallList(self.model42.gl_list)
        glPopMatrix();

    def Axis6_1(self):
        glPushMatrix();
        glScalef(0.8,0.8,0.8)
        glTranslatef(0,-1.617,0.284)
        glCallList(self.model3.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-1.012,0.227)
        glRotatef(angles[5]*12.88, 0, 1, 0);        
        glCallList(self.model32.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.698,0.145)
        glRotatef(-16.22, 1, 0, 0);
        glRotatef(angles[5]*12.88, 0, 1, 0);        
        glCallList(self.model33.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.368,0.071)
        glRotatef(angles[5]*12.88, 0, 1, 0);        
        glCallList(self.model34.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.235,0.071)
        glRotatef(angles[5]*12.88+7, 0, 1, 0);        
        glCallList(self.model35.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.234,0)
        glRotatef(-angles[5]*7.66666666667, 0, 1, 0);        
        glCallList(self.model36.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,-0.09,0)
        glRotatef(-angles[5]*7.66666666667, 0, 1, 0);        
        glCallList(self.model44.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0,0,0)
        glRotatef(90, 0, 0, 1);
        glRotatef(-angles[5]*7.66666666667, 1, 0, 0);        
        glCallList(self.model39.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.149,0.106,0)       
        glCallList(self.model45.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.219,0.018,-0.001)
        glCallList(self.model40.gl_list)
        glPopMatrix();
        
    def Axis6_2(self):
        glPushMatrix();
        glTranslatef(-0.222,0.36,0)
        glRotatef(90, 0, 0, 1);
        glRotatef(angles[5]*7.66666666667, 0, 1, 0);        
        glCallList(self.model38.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.075,0.36,0)
        glRotatef(-angles[5]*7.66666666667, 1, 0, 0);        
        glCallList(self.model46.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.035,0.44,0)
        glRotatef(-angles[5]*7.66666666667+7, 0, 1, 0);        
        glCallList(self.model47.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.079,0.646,0)
        glRotatef(angles[5]*2.3+7, 0, 1, 0);        
        glCallList(self.model48.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.009,0.646,0)
        glRotatef(angles[5]*2.3+7, 0, 1, 0);        
        glCallList(self.model48.gl_list)
        glPopMatrix();
        glPushMatrix();
        glTranslatef(-0.035,0.646,0)
        glRotatef(angles[5]+7, 0, 1, 0);        
        glCallList(self.model49.gl_list)
        glPopMatrix();

    def rotation(self, period=15):#estados de la rotacion automatica de todo el modelo ON/OFF
        #rotation timer, rotate= estado de rotacion segun la tecla r, pausetime= tiempo que demora pausada la rotacion
        #resumetime= estado actual de rotacion, angle= angulo de rotacion, period= tiempo que dura en dar una vuelta
        if self.rotate==1:
            if self.pausetime==0:
                self.resumetime=time.time()-self.starttime
            else:
                self.resumetime=time.time()-self.pausetime + self.starttime

            self.angle=(((self.resumetime)%period)/period)* 360
        elif self.rotate==0:
            self.pausetime=time.time()
            self.starttime=self.resumetime 

    def keyBoardListener(self, key, x, y):

        key = key.decode('utf-8')
        

        if key=='q':
            self.model_posx += 0.001
        elif key=='a':
            self.model_posx -= 0.001
        elif key=='w':
            self.model_posy += 0.001
        elif key=='s':
            self.model_posy -= 0.001
        elif key=='e':
            self.model_posz += 0.001
        elif key=='d':
            self.model_posz -= 0.001

        print("x=", self.model_posx, "y=", self.model_posy, "z=", self.model_posz)

# definiciones Interfaz -------------------------------------------------------------------------------------

    def Opacity(self):
        if self.ui.ButtOpac.isChecked(): 
            glEnable(GL_BLEND)           
            
        else:
            glDisable(GL_BLEND)
             

    def camara(self):

        if self.ui.radioButton.isChecked():            
          
            self.webcam = cv2.VideoCapture(1) 
        else:
            self.webcam = cv2.VideoCapture(0)

    def nextdef (self):

        if self.selectmodel < 78:
            self.selectmodel += 1
            self.Sel_Model()
            self.ui.tableWidget.setCurrentCell(self.selectmodel,0)
    
    def atrasdef (self):

        if self.selectmodel> 0:
            self.selectmodel -= 1
            self.Sel_Model()  
            self.ui.tableWidget.setCurrentCell(self.selectmodel,0)

    def Brotacion (self):
        
        if self.rotate == 1:
            self.rotate = 0
            self.rotation()
        elif self.rotate == 0:
            self.rotate = 1
            self.rotation()

    def scalslid (self):           
        self.model_scale = 0.03 + self.ui.labSlider.value()/1000

    def posslid (self):           
        self.model_posy =self.ui.labSlider2.value()/50

    def lista(self):
        item = self.ui.tableWidget.currentRow()
        self.selectmodel = item 
        self.Sel_Model()

    def iconanimate(self):
        if self.animate==1:
            iconanim = QPixmap("./Logos/Logoanim.PNG").scaled(41,41)
        else:
            iconanim = QPixmap("./Logos/Logoanimx.PNG").scaled(41,41)
        self.ui.ImgAxis.setPixmap(iconanim)

    def Sel_Model (self):

        Imante = QPixmap("./models/Images/"+str(self.selectmodel-1)+".PNG").scaled(100,100)
        Imgnext = QPixmap("./models/Images/"+str(self.selectmodel+1)+".PNG").scaled(100,100)
        Imgact = QPixmap("./models/Images/"+str(self.selectmodel)+".PNG").scaled(132,132)
        self.ui.ImgS.setPixmap(Imgnext)
        self.ui.ImgA.setPixmap(Imante)
        self.ui.Img.setPixmap(Imgact)
        #SR
        if self.selectmodel==0:
            self.model = None
        if self.selectmodel==5 or self.selectmodel==12 or self.selectmodel==24 or self.selectmodel==37 or self.selectmodel==47 or self.selectmodel==63:
            self.model =self.model3
        if self.selectmodel==1:
            self.model = getattr(self,'model%d'%self.selectmodel)
        elif 2<=self.selectmodel<=3:
            self.model = self.model2
        elif 6 <= self.selectmodel <=9:
            self.model = getattr(self,'model%d'%(self.selectmodel-2))
        elif self.selectmodel ==10:
            self.model = getattr(self,'model%d'%(self.selectmodel-3))
        elif 13 <= self.selectmodel <=21:
            self.model = getattr(self,'model%d'%(self.selectmodel-5))
        elif self.selectmodel ==22:
            self.model = getattr(self,'model%d'%(self.selectmodel-6))
        elif 25 <= self.selectmodel <=27:
            self.model = getattr(self,'model%d'%(self.selectmodel-8))
        elif 28 <= self.selectmodel <=29:
            self.model = getattr(self,'model%d'%(self.selectmodel-17))
        elif 30 <= self.selectmodel <=34:
            self.model = getattr(self,'model%d'%(self.selectmodel-10))
        elif self.selectmodel ==35:
            self.model = getattr(self,'model%d'%(self.selectmodel-11))
        elif 38 <= self.selectmodel <=44:
            self.model = getattr(self,'model%d'%(self.selectmodel-13))
        elif self.selectmodel ==45:
            self.model = getattr(self,'model%d'%(self.selectmodel-14))
        elif self.selectmodel ==48 or self.selectmodel ==64: #carcasa eje 4,5,6
            self.model = self.model25
        elif 49 <= self.selectmodel <=60:
            self.model = getattr(self,'model%d'%(self.selectmodel-17))
        elif self.selectmodel ==61:
            self.model = getattr(self,'model%d'%(self.selectmodel-18))
        elif 65 <= self.selectmodel <= 69 or 71 <= self.selectmodel <= 73:
            self.model = getattr(self,'model%d'%(self.selectmodel-33))
        elif self.selectmodel ==70: #carcasa eje 4,5,6
            self.model = self.model44
        elif 74 <= self.selectmodel <= 78:
            self.model = getattr(self,'model%d'%(self.selectmodel-29))

    def main(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(900, 680)
        glutInitWindowPosition(380, 0)
        self.window_id = glutCreateWindow(b"AR Didactic Tool")
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        self.initOpengl(640, 480)
        glutMainLoop()


    def cerrar(self):
        self.close()
        glutDestroyWindow(self.window_id)

    def minimize(self):
        self.showMinimized()

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.timer.start(20)

    def resetanim(self):
        global angles
        self.ui.S1.setValue(0)
        self.ui.S2.setValue(0)
        self.ui.S3.setValue(0)
        self.ui.S4.setValue(0)
        self.ui.S5.setValue(0)
        self.ui.S6.setValue(0)
        angles=[0,0,0,0,0,0]  

    def control_animacion(self):
        self.ui.S1.valueChanged.connect(self.slider_animacion)
        self.ui.S2.valueChanged.connect(self.slider_animacion)
        self.ui.S3.valueChanged.connect(self.slider_animacion)
        self.ui.S4.valueChanged.connect(self.slider_animacion)
        self.ui.S5.valueChanged.connect(self.slider_animacion)
        self.ui.S6.valueChanged.connect(self.slider_animacion)

    def slider_animacion(self):

        global angles

        angles[0] = -(self.ui.S1.value()/100)
        angles[1] = -(self.ui.S2.value()/100)
        angles[2] = -(self.ui.S3.value()/100)
        angles[3] = -(self.ui.S4.value()/100)
        angles[4] = -(self.ui.S5.value()/100)
        angles[5] = -(self.ui.S6.value()/100)

        for Sliders in range(1, 7):
            if Sliders == 1:
                grad=185
            elif Sliders ==2:
                grad=110
            elif Sliders ==3:
                grad=184
            elif Sliders ==4:
                grad=350
            elif Sliders ==5:
                grad=118
            elif Sliders ==6:
                grad=350

            labelvar=getattr(self.ui,'labelPercentage%d'%Sliders)
            circProg=getattr(self.ui,'circularProgress%d'%Sliders)
            slider = getattr(self.ui,'S%d'%Sliders)
            
            htmlText = """<head/><body><p align="center"><span style=" font-size:10pt; font-weight:600; ">{VALUE}</span><span style=" font-size:10pt; font-weight:600; vertical-align:super;">°</span></p>"""
            labelvar.setText(htmlText.replace("{VALUE}", str((slider.value())/100)))
            
            styleSheet = """
            QFrame{
                border-radius: 45px;
                background-color: qconicalgradient(cx:0.5, cy:0.5, angle:270, stop:{STOP_1} rgba(255,255,255,255), stop:{STOP_2} rgba(85,255,127,255));
            }
            """
            value=grad+(slider.value()/100)
            progress = ((grad*2)-value)/(grad*2)
            # GET NEW VALUES
            stop_1 = str(progress - 0.001)
            stop_2 = str(progress)
            # SETVALUES  TO NEW STYLESHEET
            newStylesheet = styleSheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2)
            # APPLY STYLESHEET WITH NEW VALUES
            circProg.setStyleSheet(newStylesheet)

    def progress(self):
        #enlaca timer con la barra de progeso
        global counter
        if counter > 100:
            self.ui.labelcarga.setText("") 
            counter=100
        #cierra la ventana de inico al termiar conteo y se abre la App 
        self.ui.progressBar.setValue(counter)  


if __name__ == '__main__':

    app = QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = OpenGLGlyphs()
    mainWindow.show()
    sys.exit(app.exec_())