import os
import base64
import cv2
import shutil
import numpy.core.multiarray
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtSql
from PIL import Image
from PyQt5.QtCore import QFile

from db import database




class NewDocWindow(QtWidgets.QWidget):
    def __init__(self, new_doc_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_doc_cb = new_doc_cb
        self.setWindowTitle("Add New Image")

        self.settings = NewDocOptions(self.new_doc_cb)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.settings)
        self.setLayout(layout)


class NewDocOptions(QtWidgets.QWidget):
    def __init__(self, new_doc_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_doc_cb = new_doc_cb

        self.choosefilebutton = QtWidgets.QPushButton("Add Files")
        self.choosefilebutton.clicked.connect(self.chooseFilesClicked)

        self.removefilebutton = QtWidgets.QPushButton("Remove Files")
        self.removefilebutton.clicked.connect(self.removeFilesClicked)

        self.filenameslabel = QtWidgets.QLabel("Files Chosen: ")
        self.listwidget = QtWidgets.QListWidget()
        self.listwidget.setDragEnabled(True)
        self.listwidget.setAcceptDrops(True)
        self.listwidget.setDropIndicatorShown(True)
        self.listwidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listwidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)

        self.submit = QtWidgets.QPushButton("Process Document")
        self.submit.clicked.connect(self.processFilesClicked)

        self.options = QtWidgets.QGroupBox("Options")
        self.namelabel = QtWidgets.QLabel("Submission Name:")
        self.name = QtWidgets.QLineEdit()
        self.imagePreprocessing = QtWidgets.QLabel("Image Preprocess:")
        self.box = QtWidgets.QCheckBox("Grayscale")
        self.box2 = QtWidgets.QCheckBox("Face Only")
        # self.box3 = QtWidgets.QCheckBox("Add to DB")
        self.infobutton = QtWidgets.QPushButton()
        self.infobutton.setIcon(QtGui.QIcon("info_icon.png"))
        self.infobutton.clicked.connect(self.displayinfo)
        optionslayout = QtWidgets.QVBoxLayout()
        optionslayout.addWidget(self.namelabel)
        optionslayout.addWidget(self.name)
        optionslayout.addWidget(self.imagePreprocessing)
        optionslayout.addWidget(self.box)
        optionslayout.addWidget(self.box2)
        # optionslayout.addWidget(self.box3)
        optionslayout.addWidget(self.infobutton, alignment=QtCore.Qt.AlignRight)
        
        self.options.setLayout(optionslayout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.choosefilebutton)
        layout.addWidget(self.removefilebutton)
        layout.addWidget(self.filenameslabel)
        layout.addWidget(self.listwidget)
        layout.addWidget(self.options)
        layout.addWidget(self.submit, alignment=QtCore.Qt.AlignBottom)
        self.setLayout(layout)


    def chooseFilesClicked(self):
        # file_dialog = QtWidgets.QFileDialog(self.centralwidget)
        # file_dialog = QtWidgets.QFileDialog(MainWindow)
        # MainWindow -> Any = QtWidgets.QMainWindow(), ^both work
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(
            "Images (*.png *.jpg *.jpeg);;PDF Files (*.pdf)")
        file_dialog.selectNameFilter("Images (*.png *.jpg *.jpeg)")
        file_names = []
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
        # Insert into list widget
        itemsTextList = [self.listwidget.item(i).text() for i in range(self.listwidget.count())]
        for file_name in file_names:
            if file_name not in itemsTextList:
                self.listwidget.insertItem(self.listwidget.count(), file_name)
                itemsTextList.append(file_name)
            else:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Please do not insert duplicates.")
                msg.setWindowTitle("Error")
                msg.exec_()

    def removeFilesClicked(self):
        items = self.listwidget.selectedItems()
        if len(items) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("No files were selected for removal.")
            msg.setInformativeText(
                'Select one or more files in the files chosen list and try again.')
            msg.setWindowTitle("Info")
            msg.exec_()
        else:
            for item in items:
                self.listwidget.takeItem(self.listwidget.row(item))

    def processFilesClicked(self):
        file_names = []
        for index in range(self.listwidget.count()):
            file_names.append(self.listwidget.item(index).text())

        # https://stackoverflow.com/questions/42511694/pyqt-how-to-get-the-row-values-from-a-select-query
        # https://stackoverflow.com/questions/46566517/turn-database-column-into-python-list-using-pyside

        query = QtSql.QSqlQuery()
        query.exec_("Select submissionname FROM submission")

        while query.next():
            if (query.value(0) == self.name.text()):
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Document names must be unique and non empty.")
                msg.setInformativeText(
                    'There is already a document with that name.')
                msg.setWindowTitle("Error")
                msg.exec_()
                return

        if len(self.name.text()) == 0:  # or query shows already exist
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Document names must be unique and non empty.")
            msg.setInformativeText(
                'Please enter a non-empty document name.')
            msg.setWindowTitle("Error")
            msg.exec_()
        elif len(file_names) == 0:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("No files were selected as part of the document.")
            msg.setInformativeText(
                'Please select files to process.')
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            # https://stackoverflow.com/questions/51306683/pyqt5-and-persistent-db-with-qtsql
            # https://stackoverflow.com/questions/50133693/how-do-i-upload-an-image-to-sqllite-database-using-pyqt-qfiledialog
            # https://stackoverflow.com/questions/32339459/python-sqllite-auto-increment-still-asking-for-the-id-field-on-insert

            for i in range(0, len(file_names)):
                # pic = QFile(file_names[i]).readAll()
                #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
                blob = file_names[i]
                #animeface
                print(blob)
                if self.box2.isChecked():
                    print("AnimeFace")
                    self.detect(blob)
                    blob = self.toBlob("animeface.jpg")
                    #os.remove("animeface.png")
                #greyscale
                if self.box.isChecked():
                    print("Greyscale")
                    if os.path.exists("animeface.jpg"):
                        img = Image.open("animeface.jpg").convert('LA')
                        img.save('greyscale.png')
                        blob = self.toBlob('greyscale.png')
                    else:
                        img = Image.open(blob).convert('LA')
                        img.save('greyscale.png')
                        blob = self.toBlob('greyscale.png')
                        #os.remove("greyscale.png")
                if os.path.exists("animeface.jpg"):
                    os.remove("animeface.jpg")
                if os.path.exists("greyscale.png"):
                    os.remove("greyscale.png")
                if not self.box2.isChecked() and not self.box.isChecked():
                    blob = self.toBlob(blob)


                # use string format instead like 'SELECT * FROM %s' % str(room)
                firstvar = "\'" + self.name.text() + "\'"
                secondvar = "\'" + os.path.basename(file_names[i]) + "\'"
                thirdvar = "\'" + blob.decode('ascii') + "\'"
                # secondvar = ",\'"+os.path.basename(file_names[i])+"\')"
                # pic2 = ",\'"+ pic +
                # string ="insert into submission values (NULL "+ firstvar+secondvar
                string = "INSERT INTO submission VALUES (NULL,%s,%s,%s)" % (firstvar, secondvar, thirdvar)
                print(string)
                query.exec_("INSERT INTO submission VALUES (NULL,%s,%s,%s)" % (firstvar, secondvar, thirdvar))

    def displayinfo(self):
        text_file = QtWidgets.QTextBrowser()
        current_working_directory = os.path.dirname(os.path.realpath(__file__))
        text = open(os.path.join(current_working_directory, "displayinfo.txt")).read()
        text_file.setText(text)
        font = QtGui.QFont()
        font.setPointSize(10)
        text_file.setFont(font)
        dialog = QtWidgets.QDialog(parent=self)

        dialog.resize(500, 500)

        temp_layout = QtWidgets.QHBoxLayout()
        temp_layout.addWidget(text_file)
        dialog.setWindowTitle("Information")
        dialog.setLayout(temp_layout)
        dialog.show()

    def toBlob(self, fname):
        with open(fname, 'rb') as f:
            blob = base64.b64encode(f.read())
            return blob

    #https://github.com/nagadomi/lbpcascade_animeface

    def detect(self, filename, cascade_file="lbpcascade_animeface.xml"):
        if not os.path.isfile(cascade_file):
            raise RuntimeError("%s: not found" % cascade_file)

        cascade = cv2.CascadeClassifier(cascade_file)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)



        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(24, 24))

        crop_img = None
        found = False
        for (x, y, w, h) in faces:
            found = True
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            crop_img = image[y:y + h, x:x + w]
        if not found:
            print("Failed to find anime face")
            shutil.copy(filename,"animeface.jpg")
            #cv2.imwrite("animeface.jpg", filename[0])
            return
        cv2.imwrite("animeface.jpg", crop_img)
