#include "Base/base.h"
#include "VideoProcessor/UI/MainWindow.h"
#include "Qt_Ext/Qt_Ext.h"



//! Qt's main function, initializes Qt Windows support
extern void qWinMain(HINSTANCE, HINSTANCE, LPSTR, int, int &, QVector<char *> &);

//! Entry point for our Windows application
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR, int nCmdShow) {

	// This code is taken from qtmain_win.cpp so we don't need to link qtmain.lib/qtmaind.lib. This way we can compile with our own choice
	// of CRT (static or dynamic) without worrying about matching Qt's, since all of Qt's code is running in a DLL
	QByteArray cmdParam= QString::fromWCharArray(GetCommandLine()).toLocal8Bit();
	int argc= 0;
   QVector<char*> argv(8);
   qWinMain( hInstance, hPrevInstance, cmdParam.data(), nCmdShow, argc, argv );

	// Create our Qt application
	//int argc = 0;
	//char** argv = nullptr;
	//QApplication app(argc, argv);
	QApplication app( argc, argv.data() );

	// Create our main window and show it
	UI::MainWindow mainWindow;
	mainWindow.show();

	// Now run our event loop
	return app.exec();
} // end WinMain()
