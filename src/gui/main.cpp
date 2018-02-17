// --------------------------------------------------------------------------------------------------
// Source code for the paper no.36 "Bayesian Collaborative Denoising
// for Monte-Carlo Rendering", submitted to the ACM SIGGRAPH 2016
// conference technical paper program.
//
// For review only. Please, do not distribute outside the review context.
//
// Final open source licence and comments to be inserted if published.
//
// Copyright(C) 2014-2016
// The Authors
//
// All rights reserved.
// ---------------------------------------------------------------------------------------------------


#include "GuiWindow.h"



using namespace std;
using namespace bcd;



#include <nanogui/nanogui.h>
#include <iostream>

using namespace nanogui;

enum test_enum {
    Item1 = 0,
    Item2,
    Item3
};

bool bvar = true;
int ivar = 12345678;
double dvar = 3.1415926;
float fvar = (float)dvar;
std::string strval = "A string";
test_enum enumval = Item2;
Color colval(0.5f, 0.5f, 0.7f, 1.f);

int tmpTest() {
	nanogui::init();

	/* scoped variables */ {
		bool use_gl_4_1 = false;// Set to true to create an OpenGL 4.1 context.
		Screen *screen = nullptr;

		if (use_gl_4_1) {
			// NanoGUI presents many options for you to utilize at your discretion.
			// See include/nanogui/screen.h for what all of these represent.
			screen = new Screen(Vector2i(500, 700), "NanoGUI test [GL 4.1]",
								/*resizable*/true, /*fullscreen*/false, /*colorBits*/8,
								/*alphaBits*/8, /*depthBits*/24, /*stencilBits*/8,
								/*nSamples*/0, /*glMajor*/4, /*glMinor*/1);
		} else {
			screen = new Screen(Vector2i(500, 700), "NanoGUI test");
		}

		bool enabled = true;
		FormHelper *gui = new FormHelper(screen);
		nanogui::ref<Window> window = gui->addWindow(Eigen::Vector2i(10, 10), "Form helper example");
		gui->addGroup("Basic types");
		gui->addVariable("bool", bvar);
		gui->addVariable("string", strval);

		gui->addGroup("Validating fields");
		gui->addVariable("int", ivar)->setSpinnable(true);
		gui->addVariable("float", fvar);
		gui->addVariable("double", dvar)->setSpinnable(true);

		gui->addGroup("Complex types");
		gui->addVariable("Enumeration", enumval, enabled)
		   ->setItems({"Item 1", "Item 2", "Item 3"});
		gui->addVariable("Color", colval);

		gui->addGroup("Other widgets");
		gui->addButton("A button", []() { std::cout << "Button pressed." << std::endl; });

		screen->setVisible(true);
		screen->performLayout();
		window->center();

		nanogui::mainloop();
	}

	nanogui::shutdown();
	return 0;
}


void launchGui()
{
	nanogui::init();
	{
		nanogui::ref<GuiWindow> window = new GuiWindow();
		window->displayUntilClosed();
	}
	nanogui::shutdown();
}

int main(int argc, char** argv)
{
	launchGui();
	//tmpTest();
	return 0;
}
