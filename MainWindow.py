from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from AddImageWindow import NewDocWindow, NewDocOptions
from db import createDB, database
from PyQt5 import QtSql
import sys
import PIL

import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from dbview import viewtable

sys.path.append("AnimeFaceNotebooks")
from AnimeFaceNotebooks import PixelateNmask as pix
from PIL.ImageQt import ImageQt
import base64
import shutil
import os

class ScrollLabel(QScrollArea):
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)

        # making widget resizable
        self.setWidgetResizable(True)

        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QVBoxLayout(content)

        # creating label
        self.label = QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

        # the setText method

    def setText(self, text):
        # setting text to the label
        self.label.setText(text)

#https://stackoverflow.com/questions/36768033/pyqt-how-to-open-new-window
class PixelateMaskWindow(QtWidgets.QMainWindow):
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(PixelateMaskWindow, self).__init__(parent)
        self.setWindowTitle("Input Tag Name")
        self.setMinimumSize(QtCore.QSize(200, 100))
        self.name = QtWidgets.QLineEdit(self)
        self.name.resize(200, 32)
        self.choosetagbutton = QtWidgets.QPushButton("Submit", self)
        self.choosetagbutton.move(0, 32)
        self.tagName = ""
        #https://stackoverflow.com/questions/55831889/sending-data-from-child-to-parent-window-in-pyqt5
        #QtCore.QObject.connect(self.choosetagbutton, self.clicked, self.choosetagbutton_Clicked)
        self.choosetagbutton.clicked.connect(self.choosetagbutton_Clicked)
        self.choosetagbutton.clicked.connect(self.clicked)

    def choosetagbutton_Clicked (self):
        self.tagName = self.name.text()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1125, 898)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 1001, 101))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(90, 190, 601, 521))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.photo.setFont(font)
        self.photo.setAlignment(QtCore.Qt.AlignCenter)
        self.photo.setObjectName("photo")
        self.photo.setScaledContents(True)
        self.loadNewImage = QtWidgets.QPushButton(self.centralwidget)
        self.loadNewImage.setGeometry(QtCore.QRect(60, 130, 201, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadNewImage.sizePolicy().hasHeightForWidth())
        self.loadNewImage.setSizePolicy(sizePolicy)
        self.submit = QtWidgets.QPushButton(self.centralwidget)
        self.submit.setGeometry(QtCore.QRect(710, 130, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.submit.setFont(font)
        self.submit.setObjectName("submit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(340, 130, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(440, 130, 121, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setFont(font)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(570, 130, 121, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit_2.setFont(font)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadNewImage.setFont(font)
        self.loadNewImage.setObjectName("loadNewImage")
        self.mask = QtWidgets.QPushButton(self.centralwidget)
        self.mask.setGeometry(QtCore.QRect(210, 770, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.mask.setFont(font)
        self.mask.setObjectName("mask")
        self.tagOutput = ScrollLabel(self.centralwidget)
        self.tagOutput.resize(600, 400)
        #self.tagOutput = QtWidgets.QLabel(self.centralwidget)
        self.tagOutput.setGeometry(QtCore.QRect(710, 190, 400, 300))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tagOutput.setFont(font)
        self.tagOutput.setAlignment(QtCore.Qt.AlignCenter)
        self.tagOutput.setObjectName("tagOutput")
        self.pixelate = QtWidgets.QPushButton(self.centralwidget)
        self.pixelate.setGeometry(QtCore.QRect(440, 770, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pixelate.setFont(font)
        self.pixelate.setObjectName("pixelate")
        self.stylegan = QtWidgets.QPushButton(self.centralwidget)
        self.stylegan.setGeometry(QtCore.QRect(660, 770, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.stylegan.setFont(font)
        self.stylegan.setObjectName("stylegan")
        #self.help = QtWidgets.QPushButton(self.centralwidget)
        #self.help.setGeometry(QtCore.QRect(950, 770, 80, 51))
        #self.help.setObjectName("help")
        #self.help.setFont(font)
        self.saveTagOutput = QtWidgets.QPushButton(self.centralwidget)
        self.saveTagOutput.setGeometry(QtCore.QRect(725, 500, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.saveTagOutput.setFont(font)
        self.saveTagOutput.setObjectName("saveTagOutput")
        self.saveImgOutput = QtWidgets.QPushButton(self.centralwidget)
        self.saveImgOutput.setGeometry(QtCore.QRect(915, 500, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.saveImgOutput.setFont(font)
        self.saveImgOutput.setObjectName("saveImgOutput")
        self.viewDatabase = QtWidgets.QPushButton(self.centralwidget)
        self.viewDatabase.setGeometry(QtCore.QRect(820, 130, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.viewDatabase.setFont(font)
        self.viewDatabase.setObjectName("viewDatabase")
        self.purePixmap = None
        #self.imageTag = None
        #https://stackoverflow.com/questions/36768033/pyqt-how-to-open-new-window
        self.pixmask = PixelateMaskWindow(self.centralwidget)
        #pyqt signal is clicked
        self.pixmask.clicked.connect(self.func)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #button methods
        self.loadNewImage.clicked.connect(self.displayNewDocumentButton_clicked)
        self.submit.clicked.connect(self.submitButton_clicked)
        self.saveTagOutput.clicked.connect(self.saveTagOutputsButton_clicked)
        self.saveImgOutput.clicked.connect(self.saveImgOutputsButton_clicked)
        self.viewDatabase.clicked.connect(self.viewDatabaseButton_clicked)
        self.pixelate.clicked.connect(self.pixelatemaskButton_clicked)
        self.mask.clicked.connect(self.maskButton_clicked)

        #hide buttons
        self.mask.hide()
        self.stylegan.hide()

    def pixelatemaskButton_clicked(self):
        self.pixmask.show()

    def maskButton_clicked(self):
        print("hello")


    def func(self):
        self.pixmask.hide()
        print(self.pixmask.tagName)
        image_fullres, image = pix.read_image("output.jpg")
        tags = pix.decode_tags(image)
        gradcam_maps = dict(zip(list(tags.keys()), pix.gradcam(image, list(tags.keys()), batch_size=5, verbose=True)))
        gradcam_maps_processed = dict(zip(gradcam_maps.keys(), list(map(pix.postprocess_grads, gradcam_maps.values()))))
        tags_to_mask = [self.pixmask.tagName]
        print(tags_to_mask)
        img = pix.mask_fullres_image(image_fullres=image_fullres, image_array=image,
                                     gradcam_maps_processed=gradcam_maps_processed,
                                     tags_to_mask=tags_to_mask)
        img.save("output.jpg", "JPEG")

        self.purePixmap = QtGui.QPixmap("output.jpg")
        self.photo.setPixmap(self.purePixmap)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Anime Illustration Recognizer"))
        self.label.setText(_translate("MainWindow", "Welcome to Anime Illustration Recognizer"))
        self.photo.setText(_translate("MainWindow", "Image Preview"))
        self.loadNewImage.setText(_translate("MainWindow", "Load a new image"))
        self.mask.setText(_translate("MainWindow", "Mask"))
        self.tagOutput.setText(_translate("MainWindow", "Tag Output "))
        self.pixelate.setText(_translate("MainWindow", "Pixelate"))
        self.stylegan.setText(_translate("MainWindow", "Stlylegan"))
        #self.help.setText(_translate("MainWindow", "?"))
        self.saveTagOutput.setText(_translate("MainWindow", "Save Tag Output"))
        self.saveImgOutput.setText(_translate("MainWindow", "Save Image Output"))
        self.viewDatabase.setText(_translate("MainWindow", "View Database"))
        self.submit.setText(_translate("MainWindow", "Submit"))
        self.label_2.setText(_translate("MainWindow", "Filename:"))


    def displayNewDocumentButton_clicked(self):
        self.new_doc_window = NewDocWindow(self)
        self.new_doc_window.show()

    def toImageBlob(self, blob):
        #change text into blobdata
        #blob2 = blob.encode("utf-8")
        #blob2 = base64.b64encode(blob)
        blob = base64.decodebytes(blob.encode('ascii'))
        #blob = base64.b64decode(blob)
        #with open("imageToSave.png", "wb") as fh:
            #fh.write(base64.decodebytes(blob))
        return blob

    def submitButton_clicked(self):
        # file_dialog = QtWidgets.QFileDialog(self.centralwidget)
        # file_dialog = QtWidgets.QFileDialog(MainWindow)
        # MainWindow -> Any = QtWidgets.QMainWindow(), ^both work
        query = QtSql.QSqlQuery()
        query.exec_("Select * FROM submission")
        blob = None
        found = False
        while query.next():
            #need to save variables/do actions while in here
            if self.lineEdit.text() == query.value(1) and self.lineEdit_2.text() == query.value(2):
                #print(query.value(2))
                #print(query.value(3))
                blob = self.toImageBlob(query.value(3))
                found = True
            else:
                if not self.lineEdit.text()  and self.lineEdit_2.text() == query.value(2):
                    #print(query.value(2))
                    #print(query.value(3))
                    blob = self.toImageBlob(query.value(3))
                    found = True
        if found == False:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Please input a valid submission - file name with extension.")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        qimg = QtGui.QImage.fromData(blob)
        self.purePixmap = QtGui.QPixmap.fromImage(qimg)
        self.photo.setPixmap(self.purePixmap)
        #self.photo.setPixmap(QtGui.QPixmap("imageToSave.png"))
        #^both work
        self.readTagOutput()

    def readTagOutput(self):
        #file_dialog = QtWidgets.QFileDialog()
        #file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        #file_dialog.setNameFilter("Text (*.txt);;PDF Files (*.pdf)")
        #file_dialog.selectNameFilter("Text (*.txt)")
        #text = open(file_names[0]).read()
        #self.tagOutput.setAlignment(QtCore.Qt.AlignLeft)

        #saves name.jpg to be used for pixmask
        self.purePixmap.save("output.jpg","JPG")
        image_fullres, image = pix.read_image("output.jpg")
        tags = pix.decode_tags(image)
        #self.imageTag = tags
        pix.show_tags(tags)
        text = open("output.txt", "r")
        self.tagOutput.setText(text.read())

        #if os.path.exists("test.txt"):
            #os.remove("test.txt")
        #if os.path.exists("name.jpg"):
            #os.remove("name.jpg")
        #self.photo.setPixmap(QtGui.QPixmap(file_names[0]))


    def viewDatabaseButton_clicked(self):
        viewtable(self.centralwidget)

    def saveTagOutputsButton_clicked(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDefaultSuffix('txt')
        file_name = file_dialog.getSaveFileName(filter="*.txt")
        if file_name:
            #replace bob.txt with blobfield or a different .txt
            shutil.copy("output.txt", file_name[0])
            #shutil.move("OCR.txt", file_name[0], copy_function=shutil.copy) crashes

    def saveImgOutputsButton_clicked(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file_dialog.setNameFilters([
             "Images (*.jpg)"])
        file_dialog.selectNameFilter("Images (*.jpg)")
        file_dialog.setDefaultSuffix("jpg")

        if file_dialog.exec_():
            #https://stackoverflow.com/questions/58408929/save-image-by-pyqt5
            file = file_dialog.selectedFiles()[0]
            #self.photo.pixmap().save(file, "JPG")
            self.purePixmap.save(file, "JPG")
            #^both work
            #with open(file, "wb") as fh:
            #    fh.write(self.photo)
            #shutil.copy(self.photo, file)
            #with open(file, "wb") as f:
            #    f.write("cat.jpg")


if __name__ == "__main__":
    import sys

    createDB()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())