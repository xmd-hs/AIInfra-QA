QT += core gui widgets network
CONFIG += c++17
QMAKE_CXXFLAGS += /utf-8
TEMPLATE = app
TARGET = AIInfraQA

SOURCES += main.cpp api_client.cpp mainwindow.cpp
HEADERS += api_client.h mainwindow.h

win32:RC_ICONS =
