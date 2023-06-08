import sys
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtCore
import sys
import numpy as np
import random
import torch
import model2
import torch.nn as nn
from torch.nn.functional import relu, sigmoid, leaky_relu
import numpy as np
from torch import Tensor
import torch
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import sqlite3
basedir = os.path.dirname(__file__)
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Skin Cancer Detection System"
 
        self.InitUI()

    def InitUI(self):
        self.setWindowTitle(self.title)
        self.setStyleSheet("QMainWindow {border-image: url(bg.jpg) 0 0 0 0 stretch stretch;}")
        buttonWindow1 = QPushButton('Try it', self)
        buttonWindowDoc = QPushButton('Doctor log in',self)
        buttonWindow1.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        buttonWindow1.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" +
                         "font-size: 30px;" +
                         "color: 'black';" +
                         "padding: 10px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        buttonWindow1.setGeometry(550,650,250,150)
        buttonWindowDoc.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        buttonWindowDoc.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" +
                         "font-size: 30px;" +
                         "color: 'black';" +
                         "padding: 10px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        buttonWindowDoc.setGeometry(950,650,250,150)
        l4 = QLabel(self)
        l4.setPixmap(QPixmap(os.path.join(basedir,"safety1.png")))
        l4.setGeometry(550,150,174,243)
        l5 = QLabel(self)
        l5.setPixmap(QPixmap(os.path.join(basedir,"safety2.png")))
        l5.setGeometry(750,150,174,243)
        l6 = QLabel(self)
        l6.setPixmap(QPixmap(os.path.join(basedir,"safety3.png")))
        l6.setGeometry(950,150,174,243)
        buttonWindow1.clicked.connect(self.buttonWindow1_onClick)
        buttonWindowDoc.clicked.connect(self.buttonWindowLogIn_onClick)
        self.showMaximized()

    @pyqtSlot()
    def buttonWindow1_onClick(self):
        self.statusBar().showMessage("Switched to window 1")
        self.cams = PatientWindow() 
        self.cams.show()
        self.close()
    @pyqtSlot()
    def buttonWindowLogIn_onClick(self):
        self.statusBar().showMessage("Switched to window Doc")
        self.cams = LoginPage() 
        self.cams.show()
        self.close()
class PatientWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #ADDBDA")
        self.setWindowTitle('Patient Page')
        self.loaded_model = None
        self.fname ='best_model_64.pth'
        self.imagePath = ''
        self.skinCancerModel = None
        self.error = QLabel(" ",self)
        self.error.setGeometry(500,600,300,30)
        self.error.setStyleSheet("color: 'red';" +
                                 "font-size: 20px;")
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
    
        self.backbutton = QPushButton(self)
        self.backbutton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.backbutton.setStyleSheet("*{border: 2px solid '#388798';" +
                                      "border-radius: 50px;" +
                                      "border-image: url(homeback.png);}"
                                      "*:hover{background: '#388798';}"
                                      )
        
        self.backbutton.setFixedSize(100,100)
        self.statusBar = QStatusBar(self)
        self.statusBar.setGeometry(0,750,200,100)

        warningl = QLabel(self)
        warningl.setPixmap(QPixmap(os.path.join(basedir,"warningl.png")))
        warningl.setGeometry(20,250,400,400)
        
        ##layoutV = QVBoxLayout()
        self.predButton = QPushButton(self)
        self.predButton.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.predButton.setText('Predict')
        self.predButton.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 35px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.predButton.setGeometry(750,650,250,150)

        

        self.resultL = QLabel("Result",self)
        self.resultL.setGeometry(1050,250,300,100)
        self.resultL.setStyleSheet("font-size: 50px;")
        
        self.resultLabel = QLabel("",self)
        self.resultLabel.setStyleSheet("font-size: 35px;")
        self.resultLabel.setGeometry(1050,370,200,100)
        ##grid.addWidget(self.pushButton,0,0)
        self.imButton = QPushButton(self)
        self.imButton.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.imButton.setText('Upload image')
        self.imButton.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 35px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.imButton.setGeometry(450,650,250,150)
        self.imButton.clicked.connect(self.getImagePath)
        self.predButton.clicked.connect(self.getPrediction)

        self.backbutton.clicked.connect(self.goMainWindow)
        self.im = QImage(QSize(224,224),QImage.Format_RGB32)
        self.imLabel = QLabel("",self)
        self.imLabel.setGeometry(400,100,500,500)
        self.imagel = QLabel("Image Preview",self)
        self.imagel.setGeometry(500,0,350,100)
        self.imagel.setStyleSheet("font-size: 50px;")
       
        self.showMaximized()
        if os.path.join(basedir,self.fname):
            self.loaded_model = torch.load(os.path.join(basedir,self.fname))
            self.statusBar.showMessage('Model is ready to prediction.')
        
    def getImagePath(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Image files (*.jpg)")

        imagePathh = fname[0]
       
        

        pixmap = QPixmap(imagePathh)
        if(len(imagePathh) != 0):
            pixmap = pixmap.scaled(500,500,Qt.KeepAspectRatio)
            self.imLabel.setPixmap(QPixmap(pixmap))
            self.imagePath = imagePathh   
        else:
            self.error.setText("Image not selected.")
 
    def getPrediction(self):
        transformer=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
            transforms.Normalize([0.5,0.5,0.5],
                                [0.5,0.5,0.5])
        ])

        if self.loaded_model and len(self.imagePath) != 0:
            self.skinCancerModel = model2.SkinCancer(num_classes=2)
            self.skinCancerModel.load_state_dict(self.loaded_model)
            self.skinCancerModel.eval()
            image = Image.open(self.imagePath)
            image_tensor=transformer(image).float()
            image_tensor=image_tensor.unsqueeze_(0)
            input=Variable(image_tensor)
            output=self.skinCancerModel(input)
            index = output.data.numpy().argmax()
            pred=model2.classes[index]
            self.resultLabel.setText(pred)
        else:
            self.error.setText("Please select image.")

    def goMainWindow(self):
        self.cams = Window()
        self.cams.showMaximized()
        self.close() 
        
 
class LoginPage(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background: #ADDBDA")
        self.setWindowTitle('Log in page')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))

        self.header = QLabel("Sign Up & Log In ", self)
        self.header.setStyleSheet("font-size: 45px;")
        self.header.setGeometry(600,50,400,100)

        self.username = QLabel("Please enter your username",self)
        self.password = QLabel("Please enter your password",self)
        self.username.setStyleSheet("font-size: 20px;")
        self.password.setStyleSheet("font-size: 20px;")
        self.username.setGeometry(450,430,245,100)
        self.password.setGeometry(800,430,245,100)


        self.lineEdit_password = QLineEdit(self)
        self.lineEdit_username = QLineEdit(self)
        self.lineEdit_username.setPlaceholderText('     username')
        self.lineEdit_password.setPlaceholderText('     password')
        self.lineEdit_password.setEchoMode(QLineEdit.Password)
        self.lineEdit_password.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)
        self.lineEdit_username.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)

        self.lineEdit_username.setGeometry(430,500,300,100)
        self.lineEdit_username.setStyleSheet("*{border: 2px solid '#388798';" +
                         "border-radius: 25px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 20px 0;}")
        self.lineEdit_password.setGeometry(780,500,300,100)
        self.lineEdit_password.setStyleSheet("*{border: 2px solid '#388798';" +
                         "border-radius: 25px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 20px 0;}")




        self.createAccount = QPushButton(self)
        self.createAccount.setText('Sign up')
        self.createAccount.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.createAccount.setGeometry(450,650,250,150)

        self.logInButton = QPushButton(self)
        self.logInButton.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.logInButton.setText('Log in')
        self.logInButton.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.logInButton.setGeometry(800,650,250,150)
        self.backbutton = QPushButton(self)
        self.backbutton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.backbutton.setStyleSheet("*{border: 2px solid '#388798';" +
                                      "border-radius: 50px;" +
                                      "border-image: url(homeback.png);}"
                                      "*:hover{background: '#388798';}"
                                      )
        self.error = QLabel("",self)
        self.error.setGeometry(450,600,300,30)
        self.error.setStyleSheet("color: 'red';")

        self.backbutton.setFixedSize(100,100)
        self.backbutton.clicked.connect(self.goMainWindow)
        self.logInButton.clicked.connect(self.logInFunction)
        self.createAccount.clicked.connect(self.createAccountFunction)
        self.showMaximized()
    
    def createAccountFunction(self):
        user = self.lineEdit_username.text()
        password = self.lineEdit_password.text()

        self.database_path = os.path.join(basedir,'database/database.db')
        if len(self.lineEdit_password.text()) == 0 or len(self.lineEdit_password.text()) == 0:
            self.error.setText("Please fill in all inputs.")
        else:
            conn = sqlite3.connect(self.database_path)
            cur = conn.cursor()
            query1 = "SELECT * FROM doctors WHERE name='"+user+"'"
            cur.execute(query1)
            result = cur.fetchone()
            if result:
                self.error.setText("Existing user. Please choose another username.")
            else:   
                query2 = "INSERT INTO doctors (name, password) VALUES ('"+user+"','"+password+"')"
                cur.execute(query2)
                conn.commit()
                self.error.setText("User is created. You can log in.")


    def logInFunction(self):
        user = self.lineEdit_username.text()
        password = self.lineEdit_password.text()
        self.database_path = os.path.join(basedir,'database/database.db')
        if len(self.lineEdit_password.text()) == 0 or len(self.lineEdit_password.text()) == 0:
            self.error.setText("Please fill in all inputs.")
        else:
            conn = sqlite3.connect(self.database_path)
            cur = conn.cursor()
            query = "SELECT * FROM doctors WHERE name='"+user+"'" 
            cur.execute(query)
            result = cur.fetchone()
            if result:
                query1 = 'SELECT password FROM doctors WHERE name =\''+user+"\'"
                cur.execute(query1)
                result_pass = cur.fetchone()[0]
                if result_pass == password:
                    self.buttonWindowDoc_onClick()
                else:
                    self.error.setText("Invalid username or password.")
            else:
                self.error.setText("There is no such username.")
    def goMainWindow(self):
        self.cams = Window()
        self.cams.showMaximized()
        self.close() 

    @pyqtSlot()
    def buttonWindowDoc_onClick(self):
        self.cams = WindowDoc() 
        self.cams.show()
        self.close()


class WindowDoc(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.imagePath = ""
        column_name = ['ID', 'LAB RESULT', 'PREDICTION RESULT', 'IMAGE']
        self.setStyleSheet("background: #ADDBDA")
        self.setWindowTitle('Doctor Page')
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        self.database_path = os.path.join(basedir,'database/database.db')
        self.dbConnection = ''
        self.button = QPushButton(self)
        self.button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.button.setStyleSheet("*{border: 2px solid '#388798';" +
                                      "border-radius: 50px;" +
                                      "border-image: url(homeback.png);}"
                                      "*:hover{background: '#388798';}"
                                      )
        self.button.setFixedSize(100,100)
        self.loadDatabase = QPushButton(self)
        self.loadDatabase.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.loadDatabase.setText('Load the database')
        self.loadDatabase.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 28px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.loadDatabase.setGeometry(200,675,250,150)

        self.uploadDatabase = QPushButton(self)
        self.uploadDatabase.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.uploadDatabase.setText('Upload new result')
        self.uploadDatabase.setStyleSheet("*{border: 4px solid '#388798';" +
                         "border-radius: 45px;" + 
                         "font-size: 28px;" +
                         "color: 'black';" +
                         "padding: 25px 0;}" + 
                         "*:hover{background: '#388798';}"
        )
        self.uploadDatabase.setGeometry(800,675,250,150)



        self.button.clicked.connect(self.goMainWindow)
        self.imageTable = QTableWidget(self)
        self.imageTable.resizeColumnsToContents()
        self.imageTable.setColumnCount(4)
        self.imageTable.setRowCount(100)
        self.imageTable.setGeometry(500,200,450,300)
        self.imageTable.setHorizontalHeaderLabels(column_name)
        self.imageTable.setColumnWidth(0,40)
        self.imageTable.setColumnWidth(2,160)
        self.imageTable.setColumnWidth(1,100)
        self.imageTable.setColumnWidth(3,110)
        self.readBlobData()
        self.loadDatabase.clicked.connect(self.readBlobData)
        self.im = QImage(QSize(224,224),QImage.Format_RGB32)
        self.imLabel = QLabel("",self)
        self.imprevlabel = QLabel("Image preview",self)
        self.imprevlabel.setGeometry(1030,60,200,200)
        self.imprevlabel.setStyleSheet("font-size: 30px;")
        qw = 650
        self.imLabel.setGeometry(1000,260,224,224)
        self.lineEdit_lab = QLineEdit(self)
        self.lineEdit_pred = QLineEdit(self)
        self.lineEdit_lab.setPlaceholderText('Lab result')
        self.lineEdit_pred.setPlaceholderText('Prediction result')
        self.lineEdit_lab.setGeometry(qw,550,250,100)
        self.lineEdit_lab.setStyleSheet("*{border: 2px solid '#388798';" +
                         "border-radius: 25px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 20px 0;}")
        self.lineEdit_pred.setGeometry(qw+300,550,250,100)
        self.lineEdit_pred.setStyleSheet("*{border: 2px solid '#388798';" +
                         "border-radius: 25px;" + 
                         "font-size: 23px;" +
                         "color: 'black';" +
                         "padding: 20px 0;}")
        self.lineEdit_pred.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)
        self.lineEdit_lab.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)
        self.chooseImage = QPushButton("Choose Image",self)
        self.chooseImage.setStyleSheet('background-color: rgb(0,0,255); color: #fff')
        self.chooseImage.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.chooseImage.setStyleSheet("*{border: 2px solid '#388798';" +
                                      "border-radius: 50px;}"
                                      "*:hover{background: '#388798';}"
                                      )
        
        self.chooseImage.setGeometry(qw+600,550,100,100)
        self.chooseImage.setText("Choose image")
        self.uploadDatabase.clicked.connect(self.uploadResult)
        self.chooseImage.clicked.connect(self.chooseImagee)
        self.error = QLabel("",self)
        self.error.setGeometry(1090,670,400,30)
        self.error.setStyleSheet("color: 'red';" +
                                 "font-size: 20px;")

        
        
        self.showMaximized()


    def writeToFile(self,data,filename):
        with open(filename,'wb') as file:
            file.write(data)

    def imagee(self):
        button = self.sender()
        imagPat = button.objectName()
        pixmap = QPixmap(imagPat)
        pixmap = pixmap.scaled(224,224,Qt.KeepAspectRatio)
        self.imLabel.setPixmap(QPixmap(pixmap))

    def readBlobData(self):
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            query =  """SELECT * FROM images"""
            cursor.execute(query)
            record = cursor.fetchall()
            rowCount = 0
            icon_size = QSize(224, 224)
            row_height = 224
            for row in record:
                idnum = row[0]
                predRes = row[1]
                labRes = row[2]
                im = row[3]
                saveLike = 'data' + str(idnum) + '.jpg'
                imagPat = 'uploadeddata/' + saveLike 
                with open(imagPat, 'wb') as f:
                    f.write(im)
                  
                pixmap = QPixmap()
                pixmap.loadFromData(row[3])
                imgButton = QPushButton()
                imgButton.setObjectName(imagPat)
                imgButton.setVisible(False)
                icon = QIcon(pixmap.scaled(icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                imgButton.setIcon(icon)
                self.imageTable.setItem(rowCount,0, QTableWidgetItem(str(idnum)))
                self.imageTable.item(rowCount,0).setFlags(QtCore.Qt.ItemIsEnabled)
                self.imageTable.setItem(rowCount,1, QTableWidgetItem(str(predRes)))
                self.imageTable.item(rowCount,1).setFlags(QtCore.Qt.ItemIsEnabled)
                self.imageTable.setItem(rowCount,2, QTableWidgetItem(str(labRes)))
                self.imageTable.item(rowCount,2).setFlags(QtCore.Qt.ItemIsEnabled)
                self.imageTable.setCellWidget(rowCount,3,imgButton)
                imgButton.clicked.connect(lambda: self.imagee())
                rowCount = rowCount + 1
            

            cursor.close()
        except sqlite3.Error as error:
            print("Failed.")
        finally: 
            if conn:
                conn.close()

    

    def getImageLabel(self,image):
        self.imageLabel = QLabel(self)
        self.imageLabel.setText("")
        self.imageLabel.setScaledContents(True)
        self.pixmap = QPixmap()
        self.pixmap.loadFromData(image,'jpg')
        self.imageLabel.setPixmap(pixmap)
        return self.imageLabel

    def chooseImagee(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\', "Image files (*.jpg)")
        imagePathh = fname[0]
        self.imagePath = imagePathh

    def uploadResult(self):
        labRes = self.lineEdit_lab.text()
        predRes = self.lineEdit_pred.text()
        if len(labRes) == 0 or len(self.imagePath) == 0:
            self.error.setText("Lab result and image is mandatory.")
        else:
            self.error.setText("Data is loaded.")
            image = open(self.imagePath,"rb")
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO images ("labResult","predictionResult","image") VALUES (?,?,?)""",[labRes,predRes,sqlite3.Binary(image.read())])
            conn.commit()
        

    def goMainWindow(self):
        self.cams = Window()
        self.cams.showMaximized()
        self.close()    
        
class ImgWidget1(QLabel):

    def __init__(self,imagePath,parent=None):
        super(ImgWidget1, self).__init__(parent)
        pic = QPixmap(imagePath)
        self.setPixmap(pic)

class ImgWidget2(QWidget):

    def __init__(self, imagePath,parent=None):
        super(ImgWidget2, self).__init__(parent)
        self.pic = QPixmap(imagePath)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pic)


if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=Window()
    sys.exit(app.exec_())