
#include "GuiWindow.h"

#include "Denoiser.h"
#include "MultiscaleDenoiser.h"
#include "IDenoiser.h"

#include "SpikeRemovalFilter.h"

#ifdef FOUND_CUDA
#include "CudaHistogramDistance.h"
#endif

#include "ImageIO.h"
#include "DeepImage.h"

#include "Chronometer.h"
#include "Utils.h"

#include "io_exr.h"

#include <nanogui/nanogui.h>

#include <Eigen/Dense>

#include <fstream>

#include <iostream>


using namespace std;
using namespace nanogui;
using namespace bcd;


NAMESPACE_BEGIN(nanogui)
NAMESPACE_BEGIN(detail)

template <> class FormWidget<FilePathFormVariable> : public Widget
{
public:
	FormWidget(Widget* i_pParentWidget) : Widget(i_pParentWidget)
	{
		setLayout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));
		m_pTextBox = new TextBox(this);
		m_pTextBox->setFixedSize(Vector2i(80, 25));
		m_pTextBox->setAlignment(TextBox::Alignment::Right);
		m_pBrowseButton = new Button(this, "Browse");
		m_pBrowseButton->setCallback(
				[this]()
				{
					m_pTextBox->setValue(file_dialog({ {"*", "All files"} }, true));
					m_pTextBox->callback()(m_pTextBox->value());
				});
	}
	void setValue(const FilePathFormVariable& i_rFilePath) { m_pTextBox->setValue(i_rFilePath.m_filePath); }
	void setEditable(bool i_editable) { m_pTextBox->setEditable(i_editable); m_pBrowseButton->setEnabled(i_editable); }
	FilePathFormVariable value() const { return FilePathFormVariable(m_pTextBox->value()); }

	void setCallback(const std::function<void(const FilePathFormVariable&)>& i_rCallback)
	{
		m_pTextBox->setCallback(
				[this, i_rCallback](const string& i_rValue) -> bool
				{
					i_rCallback(FilePathFormVariable(i_rValue));
					return true;
				});
	}
	void setButtonText(const string& i_rButtonText)
	{
		m_pBrowseButton->setCaption(i_rButtonText);
	}

private:
	TextBox* m_pTextBox;
	Button* m_pBrowseButton;
};



NAMESPACE_END(detail)
NAMESPACE_END(nanogui)



GuiWindow::GuiWindow() :
		Screen(
		Vector2i(1024, 1024), // size
		"BCD GUI", // title
		true, // resizable
		false, // fullscreen
		8, // colorBits
		8, // alphaBits
		24, // depthBits
		8, // stencilBits
		0, // nSamples
		4, // glMajor
		1 // glMinor
		),
		m_inputsAreLoaded(false),
		m_uColorInputImage(new Deepimf()),
		m_uNbOfSamplesInputImage(new Deepimf()),
		m_uHistInputImage(new Deepimf()),
		m_uCovInputImage(new Deepimf()),
		m_uOutputImage(new Deepimf())
{
	m_denoiserInputs.m_pColors = m_uColorInputImage.get();
	m_denoiserInputs.m_pHistograms = m_uHistInputImage.get();
	m_denoiserInputs.m_pNbOfSamples = m_uNbOfSamplesInputImage.get();
	m_denoiserInputs.m_pSampleCovariances = m_uCovInputImage.get();

	m_denoiserOutputs.m_pDenoisedColors = m_uOutputImage.get();

	cout << "GuiWindow constructed!" << endl;
}

GuiWindow::~GuiWindow()
{
}

void GuiWindow::displayUntilClosed()
{
	buildGui();

	setVisible(true);

	nanogui::mainloop();
}

void GuiWindow::drawContents()
{

}

void GuiWindow::buildGui()
{
	FormHelper* pFormHelper = new FormHelper(this);
	nanogui::ref<Window> window = pFormHelper->addWindow(Eigen::Vector2i(10, 10), "BCD Form");

	pFormHelper->addGroup("Inputs");
	auto inputColorWidget = pFormHelper->addVariable("Color image", m_colorInputFilePath);
	auto inputHistWidget = pFormHelper->addVariable("Histogram image", m_histInputFilePath);
	auto inputCovWidget = pFormHelper->addVariable("Covariance image", m_covInputFilePath);

	pFormHelper->addGroup("Parameters");
	pFormHelper->addVariable("Hist distance threshold", m_denoiserParameters.m_histogramDistanceThreshold);
	pFormHelper->addVariable("Use CUDA", m_denoiserParameters.m_useCuda);
	pFormHelper->addVariable("Nb of cores used (0 = default)", m_denoiserParameters.m_nbOfCores);
	pFormHelper->addVariable("Patch radius", m_denoiserParameters.m_patchRadius);
	pFormHelper->addVariable("Search window radius", m_denoiserParameters.m_searchWindowRadius);
	pFormHelper->addVariable("Random pixel order", m_denoiserParameters.m_useRandomPixelOrder);
	pFormHelper->addVariable("Marked pixels skipping probability", m_denoiserParameters.m_markedPixelsSkippingProbability);
	pFormHelper->addVariable("Min eigen value", m_denoiserParameters.m_minEigenValue);

	pFormHelper->addGroup("Outputs");
	pFormHelper->addVariable("Output file", m_outputFilePath);

	auto denoiseButton = pFormHelper->addButton("Denoise",
			[this]()
			{
				if(!ifstream(m_outputFilePath.m_filePath))
					if(!ofstream(m_outputFilePath.m_filePath))
					{
						cerr << "Error: invalid output path '" << m_outputFilePath.m_filePath << "'!" << endl;
						return;
					}
				Denoiser denoiser;
				denoiser.setInputs(m_denoiserInputs);
				denoiser.setParameters(m_denoiserParameters);
				denoiser.setOutputs(m_denoiserOutputs);

				denoiser.denoise();

//				checkAndPutToZeroNegativeInfNaNValues(outputDenoisedColorImage); // TODO: put in utils?

				ImageIO::writeEXR(*m_denoiserOutputs.m_pDenoisedColors, m_outputFilePath.m_filePath.c_str());
				cout << "Written denoised output in file " << m_outputFilePath.m_filePath.c_str() << endl;

			});
//	denoiseButton->setEnabled(false);

	performLayout();
	window->center();


	inputColorWidget->setCallback(
			[this](const FilePathFormVariable& i_rFilePath)
			{
				if(i_rFilePath == m_colorInputFilePath)
					return;
				if(!ifstream(i_rFilePath.m_filePath))
				{
					cerr << "Warning: file '" << i_rFilePath.m_filePath << "' does not exist" << endl;
					return;
				}
				m_colorInputFilePath = i_rFilePath;
				cout << "loading file '" << i_rFilePath.m_filePath << "'..." << endl;
				if(ImageIO::loadEXR(*m_uColorInputImage, i_rFilePath.m_filePath.c_str()))
				{
					cout << "file '" <<  i_rFilePath.m_filePath << "' loaded!" << endl;

				}
				else
					cerr << "ERROR: loading of file '" << i_rFilePath.m_filePath << "' failed!" << endl;
			}
			);

	inputHistWidget->setCallback(
			[this](const FilePathFormVariable& i_rFilePath)
			{
				if(i_rFilePath == m_histInputFilePath)
					return;
				if(!ifstream(i_rFilePath.m_filePath))
				{
					cerr << "Warning: file '" << i_rFilePath.m_filePath << "' does not exist" << endl;
					return;
				}
				m_histInputFilePath = i_rFilePath;
				cout << "loading file '" << i_rFilePath.m_filePath << "'..." << endl;
				Deepimf histAndNbOfSamplesImage;
				if(ImageIO::loadMultiChannelsEXR(histAndNbOfSamplesImage, i_rFilePath.m_filePath.c_str()))
				{
					Utils::separateNbOfSamplesFromHistogram(*m_uHistInputImage, *m_uNbOfSamplesInputImage, histAndNbOfSamplesImage);
					cout << "file '" <<  i_rFilePath.m_filePath << "' loaded!" << endl;
				}
				else
					cerr << "ERROR: loading of file '" << i_rFilePath.m_filePath << "' failed!" << endl;
			}
			);

	inputCovWidget->setCallback(
			[this](const FilePathFormVariable& i_rFilePath)
			{
				if(i_rFilePath == m_covInputFilePath)
					return;
				if(!ifstream(i_rFilePath.m_filePath))
				{
					cerr << "Warning: file '" << i_rFilePath.m_filePath << "' does not exist" << endl;
					return;
				}
				m_covInputFilePath = i_rFilePath;
				cout << "loading file '" << i_rFilePath.m_filePath << "'..." << endl;
				if(ImageIO::loadEXR(*m_uCovInputImage, i_rFilePath.m_filePath.c_str()))
					cout << "file '" <<  i_rFilePath.m_filePath << "' loaded!" << endl;
				else
					cerr << "ERROR: loading of file '" << i_rFilePath.m_filePath << "' failed!" << endl;
			}
			);



}



