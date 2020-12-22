from PyQt5 import QtSql, QtGui, QtWidgets
#https://www.tutorialspoint.com/pyqt/pyqt_database_handling.htm
database = QtSql.QSqlDatabase.addDatabase('QSQLITE')


def createDB():

    database.setDatabaseName('submissions.db')


    if not database.open():
        QtGui.QMessageBox.critical(None, QtGui.qApp.tr("Cannot open database"),
                                   QtGui.qApp.tr("Unable to establish a database connection.\n"
                                                 "This example needs SQLite support. Please read "
                                                 "the Qt SQL driver documentation for information "
                                                 "how to build it.\n\n" "Click Cancel to exit."),
                                   QtGui.QMessageBox.Cancel)

        return False

    query = QtSql.QSqlQuery()
    #query.exec_("create table submission(id INTEGER primary key AUTOINCREMENT,"
    #           "submissionname varchar(100), filename varchar(100))")
    query.exec_("create table submission(id INTEGER primary key AUTOINCREMENT,"
                "submissionname varchar(100), filename varchar(100), strblob text)")
    #query.exec_("insert into sportsmen values(1, 'Roger', 'Federer')")

    return True


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    createDB()