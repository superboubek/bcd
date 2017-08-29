

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

#include <json.hpp>

#include <nanogui/nanogui.h>

#include <Eigen/Dense>

#include <array>

#include <fstream>

#include <iostream>


using namespace std;
using namespace nanogui;
using namespace bcd;
using json = nlohmann::json;


const float DisplayView::s_zoomFactor = 1.08f;
const float DisplayView::s_wheelFactor = 1.f;


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
	void setValue(const FilePathFormVariable& i_rFilePath)
	{
		m_pTextBox->setValue(i_rFilePath.m_filePath);
	}
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
	void callback() // TODO: this method is just temporary?
	{
		m_pTextBox->callback()(m_pTextBox->value());
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
		m_nbOfScales(3),
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

void testJson()
{
	json j;
	j["pi"] = 3.141;
	j["happy"] = true;
	j["name"] = "Niels";

	cout << setw(2) << j << endl;
}

void GuiWindow::displayUntilClosed()
{
	initOpenGL();
	buildGui();

//	testJson();

	setVisible(true);

	nanogui::mainloop();
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
	pFormHelper->addVariable("Nb of scales", m_nbOfScales);
	pFormHelper->addVariable("Hist distance threshold", m_denoiserParameters.m_histogramDistanceThreshold);
	pFormHelper->addVariable("Use CUDA", m_denoiserParameters.m_useCuda);
	pFormHelper->addVariable("Nb of cores used (0 = default)", m_denoiserParameters.m_nbOfCores);
	pFormHelper->addVariable("Patch radius", m_denoiserParameters.m_patchRadius);
	pFormHelper->addVariable("Search window radius", m_denoiserParameters.m_searchWindowRadius);
	pFormHelper->addVariable("Random pixel order", m_denoiserParameters.m_useRandomPixelOrder);
	pFormHelper->addVariable("Marked pixels skipping probability", m_denoiserParameters.m_markedPixelsSkippingProbability);
	pFormHelper->addVariable("Min eigen value", m_denoiserParameters.m_minEigenValue);

	pFormHelper->addGroup("Outputs");
	auto outputFileWidget = pFormHelper->addVariable("Output file", m_outputFilePath);

	auto denoiseButton = pFormHelper->addButton("Denoise",
			[this]()
			{
				if(!ifstream(m_outputFilePath.m_filePath))
					if(!ofstream(m_outputFilePath.m_filePath))
					{
						cerr << "Error: invalid output path '" << m_outputFilePath.m_filePath << "'!" << endl;
						return;
					}

				Deepimf outputDenoisedColorImage(*m_denoiserInputs.m_pColors);
				m_denoiserOutputs.m_pDenoisedColors = &outputDenoisedColorImage;

				unique_ptr<IDenoiser> uDenoiser = nullptr;

				if(m_nbOfScales > 1)
					uDenoiser.reset(new MultiscaleDenoiser(m_nbOfScales));
				else
					uDenoiser.reset(new Denoiser());

				uDenoiser->setInputs(m_denoiserInputs);
				uDenoiser->setOutputs(m_denoiserOutputs);
				uDenoiser->setParameters(m_denoiserParameters);

				uDenoiser->denoise();

//				checkAndPutToZeroNegativeInfNaNValues(outputDenoisedColorImage); // TODO: put in utils?

				ImageIO::writeEXR(outputDenoisedColorImage, m_outputFilePath.m_filePath.c_str());
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

					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, m_textureIds[0]);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
							m_uColorInputImage->getWidth(),
							m_uColorInputImage->getHeight(),
							0, GL_RGB, GL_FLOAT,
							m_uColorInputImage->getDataPtr());

					m_displayView.reset(width(), height(), m_uColorInputImage->getWidth(), m_uColorInputImage->getHeight());
					setCamera();
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
				if(ImageIO::loadMultiChannelsEXR(*m_uCovInputImage, i_rFilePath.m_filePath.c_str()))
					cout << "file '" <<  i_rFilePath.m_filePath << "' loaded!" << endl;
				else
					cerr << "ERROR: loading of file '" << i_rFilePath.m_filePath << "' failed!" << endl;
			}
			);

	// begin of tmp adds to ease testing
	inputColorWidget->setValue(FilePathFormVariable("/data/boughida/projects/bcd/data/inputs/test.exr"));
	inputColorWidget->callback();
	inputHistWidget->setValue(FilePathFormVariable("/data/boughida/projects/bcd/data/inputs/test_hist.exr"));
	inputHistWidget->callback();
	inputCovWidget->setValue(FilePathFormVariable("/data/boughida/projects/bcd/data/inputs/test_cov.exr"));
	inputCovWidget->callback();
	outputFileWidget->setValue(FilePathFormVariable("/data/boughida/projects/bcd/data/outputs/tmp/test_BCDfiltered.exr"));
	outputFileWidget->callback();
	// end of tmp adds to ease testing

}

void GuiWindow::initTextures()
{
	const int nbOfTextures = 1;

	glGenTextures(m_textureIds.size(), m_textureIds.data());

	GLsizei width = 1, height = 1;
	array<float, 3> defaultData = { 0.6f, 0.2f, 0.1f };


	for(int i = 0; i < m_textureIds.size(); ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_textureIds[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0,
				GL_RGB, GL_FLOAT, defaultData.data());
	}
}

void GuiWindow::initOpenGL()
{

	// glEnable(GL_TEXTURE_2D);

	initTextures();
	m_shaderProgram.init(
		/* An identifying name */
		"a_simple_shader",

		/* Vertex shader */
		"#version 330\n"
		"uniform mat4 modelViewProj;\n"
		"in vec3 position;\n"
		"in vec2 vertexTexCoords;\n"
		"out vec2 texCoords;\n"
		"void main() {\n"
		"	gl_Position = modelViewProj * vec4(position, 1.0);\n"
		"	texCoords = vertexTexCoords;\n"
		"}",

		/* Fragment shader */
		"#version 330\n"
		"in vec2 texCoords;\n"
		"out vec4 color;\n"
		"uniform float intensity;\n"
		"uniform sampler2D textureSampler;\n"
		"void main() {\n"
		"	color = vec4(texture(textureSampler, texCoords).rgb, 1.0);\n"
		// "	color = vec4(texCoords.x, 0, texCoords.y, 1.0);\n"
		"}"
	);

	MatrixXu indices(3, 2); /* Draw 2 triangles */
	indices.col(0) << 0, 1, 2;
	indices.col(1) << 2, 3, 0;

	MatrixXf positions(3, 4);
	positions.col(0) << -1, -1, 0;
	positions.col(1) <<  1, -1, 0;
	positions.col(2) <<  1,  1, 0;
	positions.col(3) << -1,  1, 0;

	MatrixXf vertexTexCoords(2, 4);
	vertexTexCoords.col(0) << 0, 1;
	vertexTexCoords.col(1) << 1, 1;
	vertexTexCoords.col(2) << 1, 0;
	vertexTexCoords.col(3) << 0, 0;

	m_shaderProgram.bind();
	m_shaderProgram.uploadIndices(indices);
	m_shaderProgram.uploadAttrib("position", positions);
	m_shaderProgram.uploadAttrib("vertexTexCoords", vertexTexCoords);


//	m_shaderProgram.setUniform("intensity", 0.5f);
	m_shaderProgram.setUniform("textureSampler", GLuint(0)); // TODO: replace this 0 by a variable?
//	glUniform1i(m_shaderProgram.uniform("textureSampler"), 0);

	setCamera();
}

void GuiWindow::drawContents()
{
	using namespace nanogui;

	/* Draw the window contents using OpenGL */

	for(int i = 0; i < m_textureIds.size(); ++i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_textureIds[i]);
	}
	m_shaderProgram.bind();

//	Matrix4f mvp;
//	mvp.setIdentity();
//	Eigen::Matrix4f
//	mvp.topLeftCorner<3,3>() = Matrix3f(Eigen::AngleAxisf((float) glfwGetTime(),  Vector3f::UnitZ())) * 0.25f;
//
//	mvp.row(0) *= (float) mSize.y() / (float) mSize.x();
//
//	m_shaderProgram.setUniform("modelViewProj", mvp);

	/* Draw 2 triangles starting at index 0 */
	m_shaderProgram.drawIndexed(GL_TRIANGLES, 0, 2);
}


void GuiWindow::setCamera()
{
	m_displayView.print();

	Eigen::Matrix4f mvp;
	mvp.setIdentity();
	float zoomFactorX = 2.f / m_displayView.m_width;
	float zoomFactorY = 2.f / m_displayView.m_height;
	float dx = -1 - m_displayView.m_xMin * zoomFactorX;
	float dy = -1 - m_displayView.m_yMin * zoomFactorY;

	mvp(0,0) = zoomFactorX;
	mvp(1,1) = zoomFactorY;
	mvp(0,3) = dx;
	mvp(1,3) = dy;

	m_shaderProgram.bind();
	m_shaderProgram.setUniform("modelViewProj", mvp);
}

void DisplayView::reset(int windowWidth, int windowHeight, int imageWidth, int imageHeight)
{
	m_initialWidth = float(windowWidth) / float(imageWidth);
	m_initialHeight = float(windowHeight) / float(imageHeight);
	m_width = m_initialWidth;
	m_height = m_initialHeight;
	m_xMin = -0.5f * m_initialWidth;
	m_yMin = -0.5f * m_initialHeight;
	m_totalZoomExponent = 0.f;
}

void DisplayView::print()
{
	cout << "viewport (x,y) in [" << m_xMin << ", " << m_xMin + m_width << "] x [" << m_yMin << ", " << m_yMin + m_height << "]" << endl;
}


bool GuiWindow::mouseButtonEvent(const Vector2i &p, int button, bool down, int modifiers)
{
	cout << "Entering GuiWindow::mouseButtonEvent(p = " << p << ", button = " << button << ", down = " << down << ", modifiers = " << modifiers << ")" << endl;
	if(Widget::mouseButtonEvent(p, button, down, modifiers))
		return true;
	cout << "passed super call Widget::mouseButtonEvent" << endl;

	return false;
}


bool GuiWindow::scrollEvent(const Vector2i &p, const Vector2f &rel)
{
	cout << "Entering GuiWindow::scrollEvent(p = " << p << ", rel = " << rel << ")" << endl;
	if(Widget::scrollEvent(p, rel))
		return true;
	cout << "passed super call Widget::scrollEvent" << endl;

	int mouseX = p(0);
	int mouseY = p(1);

	cout << "mouse (x,y) = (" << mouseX << ", " << mouseY << ")" << endl;

	int nx = width();
	int ny = height();
	float x = m_displayView.m_xMin + (m_displayView.m_width * mouseX) / nx;
	float y = m_displayView.m_yMin + (m_displayView.m_height * (ny - mouseY)) / ny;
	m_displayView.m_totalZoomExponent -= rel(1) * m_displayView.s_wheelFactor;
	m_displayView.m_width = m_displayView.m_initialWidth * pow(m_displayView.s_zoomFactor, m_displayView.m_totalZoomExponent);
	m_displayView.m_height = m_displayView.m_initialHeight * pow(m_displayView.s_zoomFactor, m_displayView.m_totalZoomExponent);
	m_displayView.m_xMin = x - (m_displayView.m_width * mouseX) / nx;
	m_displayView.m_yMin = y - (m_displayView.m_height * (ny - mouseY)) / ny;

	setCamera();

//	update();


	return true;
}

