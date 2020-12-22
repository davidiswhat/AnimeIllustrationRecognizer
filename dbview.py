import sys
from PyQt5 import QtCore, QtSql, QtWidgets
#https://www.tutorialspoint.com/pyqt/pyqt_database_handling.html

def initializeModel(model):
    model.setTable('submission')
    model.setEditStrategy(QtSql.QSqlTableModel.OnFieldChange)
    model.select()
    model.setHeaderData(0, QtCore.Qt.Horizontal, "ID")
    model.setHeaderData(1, QtCore.Qt.Horizontal, "Submission Name")
    model.setHeaderData(2, QtCore.Qt.Horizontal, "File Name")
    model.setHeaderData(3, QtCore.Qt.Horizontal, "Image data")


def createView(title, model):
    view = QtWidgets.QTableView()
    view.setModel(model)
    view.setWindowTitle(title)
    return view


def addrow():
    print()
    model.rowCount()
    ret = model.insertRows(model.rowCount(), 1)
    print()
    ret

def selectrow(unit):
    print(unit)

def findrow(i):
    delrow = i.row()

def viewtable(cw):
    db = QtSql.QSqlDatabase.addDatabase('QSQLITE')
    db.setDatabaseName('submissions.db')
    model = QtSql.QSqlTableModel()
    delrow = -1
    initializeModel(model)

    view1 = createView("Table Model (View 1)", model)
    view1.clicked.connect(findrow)

    dlg = QtWidgets.QDialog(cw)
    layout = QtWidgets.QVBoxLayout(cw)
    layout.addWidget(view1)

    #button = QtWidgets.QPushButton("Add a row")
    #button.clicked.connect(addrow)
    #layout.addWidget(button)

    btn1 = QtWidgets.QPushButton("Delete a row")
    btn1.clicked.connect(lambda: model.removeRow(view1.currentIndex().row()))
    layout.addWidget(btn1)

    dlg.setLayout(layout)
    dlg.setWindowTitle("Database")
    dlg.show()