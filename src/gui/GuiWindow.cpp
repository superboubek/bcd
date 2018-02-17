

#include "GuiWindow.h"

#include "Denoiser.h"
#include "MultiscaleDenoiser.h"
#include "IDenoiser.h"

#include "SpikeRemovalFilter.h"

#ifdef FOUND_CUDA
#include "CudaHistogramDistance.h"
#endif

#include "CovarianceMatrix.h"
#include "ImageIO.h"
#include "DeepImage.h"

#include "Chronometer.h"
#include "Utils.h"

#include <nanogui/nanogui.h>

#include <Eigen/Dense>

#include <array>

#include <fstream>

#include <iostream>


using namespace std;
using namespace nanogui;
using namespace bcd;


const float ViewFrame::s_zoomFactor = 1.08f;
const float ViewFrame::s_wheelFactor = 1.f;


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
		m_pTextBox->setCallback(
				[this](const string& i_rValue) -> bool
				{
					callback()(FilePathFormVariable(i_rValue));
					return true;
				});
		m_pBrowseButton = new Button(this, "Browse");
		m_pBrowseButton->setCallback(
				[this]()
				{
					m_pTextBox->setValue(file_dialog({ {"*", "All files"} }, true));
					callback()(m_pTextBox->value());
				});
	}
	void setValue(const FilePathFormVariable& i_rFilePath)
	{
		m_pTextBox->setValue(i_rFilePath.m_filePath);
	}
	void setEditable(bool i_editable) { m_pTextBox->setEditable(i_editable); m_pBrowseButton->setEnabled(i_editable); }
	FilePathFormVariable value() const { return FilePathFormVariable(m_pTextBox->value()); }

	std::function<void(const FilePathFormVariable& str)> callback() const { return m_callback; }
	void setCallback(const std::function<void(const FilePathFormVariable& str)>& i_rCallback) { m_callback = i_rCallback; }

	void setButtonText(const string& i_rButtonText)
	{
		m_pBrowseButton->setCaption(i_rButtonText);
	}

private:

	TextBox* m_pTextBox;
	Button* m_pBrowseButton;

	std::function<void(const FilePathFormVariable& str)> m_callback;

};



NAMESPACE_END(detail)
NAMESPACE_END(nanogui)



GuiWindow::GuiWindow() :
		Screen(
		Vector2i(1024, 1024), // size
		"BCD Tweaker", // title
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
		m_uFormHelper(nullptr),
		m_uParametersSubWindow(nullptr),
		m_uDisplaySubWindow(nullptr),
		m_uDenoisingProgressBar(nullptr),
		m_hideAllSubWindows(false),
		m_inputsAreLoaded(false),

		m_uColorInputImage(new Deepimf()),
		m_uNbOfSamplesInputImage(new Deepimf()),
		m_uHistInputImage(new Deepimf()),
		m_uCovInputImage(new Deepimf()),
		m_uCovTraceInputImage(new Deepimf()),

		m_uPrefilteredColorInputImage(new Deepimf()),
		m_uPrefilteredNbOfSamplesInputImage(new Deepimf()),
		m_uPrefilteredHistInputImage(new Deepimf()),
		m_uPrefilteredCovInputImage(new Deepimf()),
		m_uPrefilteredCovTraceInputImage(new Deepimf()),

		m_uOutputImage(new Deepimf()),
		m_currentShaderProgramType(EShaderProgram::empty),
		m_pCurrentShaderProgram(nullptr),
		m_oldDisplayType(EDisplayType::count),
		m_currentDisplayType(EDisplayType::colorInput),
		m_displayChanged(true),
		m_viewChanged(true),
		m_gamma(2.2f),
		m_exposure(1.f),
		m_covTraceScale(1.f)
{
	m_denoiserInputs.m_pColors = m_uColorInputImage.get();
	m_denoiserInputs.m_pHistograms = m_uHistInputImage.get();
	m_denoiserInputs.m_pNbOfSamples = m_uNbOfSamplesInputImage.get();
	m_denoiserInputs.m_pSampleCovariances = m_uCovInputImage.get();

	m_denoiserOutputs.m_pDenoisedColors = m_uOutputImage.get();


	for(size_t i = 0; i < size_t(EDisplayType::count); ++i)
		m_displayTypeIsVisible[i] = true;

	m_viewFrame.reset(width(), height(), 512, 512);
}

GuiWindow::~GuiWindow()
{
}

void GuiWindow::displayUntilClosed()
{
	initOpenGL();
	buildGui();

	setVisible(true);

	nanogui::mainloop();
}

void GuiWindow::setTextureData(ETexture i_texture, const DeepImage<float>& i_rImage)
{
	GLint internalFormat;
	GLenum format;
	switch(i_rImage.getDepth())
	{
	case 1:
		internalFormat = GL_R32F;
		format = GL_RED;
		break;
	case 3:
		internalFormat = GL_RGBA32F;
		format = GL_RGB;
		break;
	default:
		cerr << "Error: call to setTextureData with image which depth is not 1 or 3 (depth = " << i_rImage.getDepth() << ")" << endl;
		return;
	}

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_textureIds[size_t(i_texture)]);
	glTexImage2D(
			GL_TEXTURE_2D, 0, internalFormat,
			i_rImage.getWidth(),
			i_rImage.getHeight(),
			0, format, GL_FLOAT,
			i_rImage.getDataPtr()
			);
	m_displayChanged = true;
}


void GuiWindow::loadInputsAndParameters()
{
	loadInputsAndParameters(file_dialog({ { "bcd.json", "BCD Json file" } }, true));
}

void GuiWindow::loadInputsAndParameters(const string& i_rFilePath)
{
	if(!ParametersIO::load(m_pipelineParameters, i_rFilePath, m_pipelineParametersSelector))
		return;

	loadInputColorFile(m_pipelineParameters.m_inputFileNames.m_colors);
	loadInputHistoFile(m_pipelineParameters.m_inputFileNames.m_histograms);
	loadInputCovarFile(m_pipelineParameters.m_inputFileNames.m_covariances);
}

void GuiWindow::saveInputsAndParameters()
{
	string filePath = file_dialog({ { "bcd.json", "BCD Json file" } }, true);
	if(filePath == "")
		return;

	m_pipelineParameters.m_inputFileNames.m_colors = m_colorInputFilePath.m_filePath;
	m_pipelineParameters.m_inputFileNames.m_covariances = m_covarInputFilePath.m_filePath;
	m_pipelineParameters.m_inputFileNames.m_histograms = m_histoInputFilePath.m_filePath;

	ParametersIO::write(m_pipelineParameters, filePath, m_pipelineParametersSelector);
}


void GuiWindow::loadInputColorFile(const string& i_rFilePath)
{
	if(i_rFilePath == m_colorInputFilePath.m_filePath)
		return;
	if(!ifstream(i_rFilePath))
	{
		cerr << "Warning: file '" << i_rFilePath << "' does not exist" << endl;
		return;
	}
	m_colorInputFilePath.m_filePath = i_rFilePath;
	cout << "loading file '" << i_rFilePath << "'..." << endl;
	if(ImageIO::loadEXR(*m_uColorInputImage, i_rFilePath.c_str()))
	{
		cout << "file '" <<  i_rFilePath << "' loaded!" << endl;
		setTextureData(ETexture::colorInput, *m_uColorInputImage);
	}
	else
		cerr << "ERROR: loading of file '" << i_rFilePath << "' failed!" << endl;
}

void GuiWindow::loadInputHistoFile(const string& i_rFilePath)
{
	if(i_rFilePath == m_histoInputFilePath.m_filePath)
		return;
	if(!ifstream(i_rFilePath))
	{
		cerr << "Warning: file '" << i_rFilePath << "' does not exist" << endl;
		return;
	}
	m_histoInputFilePath.m_filePath = i_rFilePath;
	cout << "loading file '" << i_rFilePath << "'..." << endl;
	Deepimf histAndNbOfSamplesImage;
	if(ImageIO::loadMultiChannelsEXR(histAndNbOfSamplesImage, i_rFilePath.c_str()))
	{
		Utils::separateNbOfSamplesFromHistogram(*m_uHistInputImage, *m_uNbOfSamplesInputImage, histAndNbOfSamplesImage);
		cout << "file '" <<  i_rFilePath << "' loaded!" << endl;
	}
	else
		cerr << "ERROR: loading of file '" << i_rFilePath << "' failed!" << endl;
}

void GuiWindow::loadInputCovarFile(const string& i_rFilePath)
{
	if(i_rFilePath == m_covarInputFilePath.m_filePath)
		return;
	if(!ifstream(i_rFilePath))
	{
		cerr << "Warning: file '" << i_rFilePath << "' does not exist" << endl;
		return;
	}
	m_covarInputFilePath.m_filePath = i_rFilePath;
	cout << "loading file '" << i_rFilePath << "'..." << endl;
	if(ImageIO::loadMultiChannelsEXR(*m_uCovInputImage, i_rFilePath.c_str()))
	{
		cout << "file '" <<  i_rFilePath << "' loaded!" << endl;

		computeCovTraceImage(*m_uCovTraceInputImage, *m_uCovInputImage);
		setTextureData(ETexture::covTraceInput, *m_uCovTraceInputImage);
	}
	else
		cerr << "ERROR: loading of file '" << i_rFilePath << "' failed!" << endl;
}

void GuiWindow::computeCovTraceImage(DeepImage<float>& o_rCovTraceImage, const DeepImage<float>& i_rCovImage)
{
	assert(i_rCovImage.getDepth() == 6);

	int w = i_rCovImage.getWidth();
	int h = i_rCovImage.getHeight();
	o_rCovTraceImage.resize(w, h, 1);
	auto covIt = i_rCovImage.begin();
	auto covItEnd = i_rCovImage.end();
	auto covTraceIt = o_rCovTraceImage.begin();
	for(; covIt != covItEnd; ++covIt, ++covTraceIt)
		covTraceIt[0] = sqrt(
				covIt[int(ESymmetricMatrix3x3Data::e_xx)] +
				covIt[int(ESymmetricMatrix3x3Data::e_yy)] +
				covIt[int(ESymmetricMatrix3x3Data::e_zz)]);
}

void GuiWindow::prefilter()
{
	*m_uPrefilteredColorInputImage = *m_uColorInputImage;
	*m_uPrefilteredNbOfSamplesInputImage = *m_uNbOfSamplesInputImage;
	*m_uPrefilteredHistInputImage = *m_uHistInputImage;
	*m_uPrefilteredCovInputImage = *m_uCovInputImage;

	SpikeRemovalFilter::filter(
			*m_uPrefilteredColorInputImage,
			*m_uPrefilteredNbOfSamplesInputImage,
			*m_uPrefilteredHistInputImage,
			*m_uPrefilteredCovInputImage,
			m_pipelineParameters.m_prefilteringParameters.m_spikeRemovalThresholdStDevFactor);

	setTextureData(ETexture::prefilteredColorInput, *m_uPrefilteredColorInputImage);

	computeCovTraceImage(*m_uPrefilteredCovTraceInputImage, *m_uPrefilteredCovInputImage);
	setTextureData(ETexture::prefilteredCovTraceInput, *m_uPrefilteredCovTraceInputImage);

	if(m_currentDisplayType == EDisplayType::covTraceInput)
		m_currentDisplayType = EDisplayType::prefilteredCovTraceInput;
	else
		m_currentDisplayType = EDisplayType::prefilteredColorInput;

}

void GuiWindow::denoise()
{

	*m_uOutputImage = *m_uColorInputImage;

	unique_ptr<IDenoiser> uDenoiser = nullptr;

	if(m_pipelineParameters.m_denoiserParameters.m_nbOfScales > 1)
		uDenoiser.reset(new MultiscaleDenoiser(m_pipelineParameters.m_denoiserParameters.m_nbOfScales));
	else
		uDenoiser.reset(new Denoiser());

	if(m_pipelineParameters.m_prefilteringParameters.m_performSpikeRemoval)
	{
		prefilter();
		m_denoiserInputs.m_pColors = m_uPrefilteredColorInputImage.get();
		m_denoiserInputs.m_pHistograms = m_uPrefilteredHistInputImage.get();
		m_denoiserInputs.m_pNbOfSamples = m_uPrefilteredNbOfSamplesInputImage.get();
		m_denoiserInputs.m_pSampleCovariances = m_uPrefilteredCovInputImage.get();
	}
	else
	{
		m_denoiserInputs.m_pColors = m_uColorInputImage.get();
		m_denoiserInputs.m_pHistograms = m_uHistInputImage.get();
		m_denoiserInputs.m_pNbOfSamples = m_uNbOfSamplesInputImage.get();
		m_denoiserInputs.m_pSampleCovariances = m_uCovInputImage.get();
	}

	uDenoiser->setInputs(m_denoiserInputs);
	uDenoiser->setOutputs(m_denoiserOutputs);
	uDenoiser->setParameters(m_pipelineParameters.m_denoiserParameters.m_monoscaleParameters);
	uDenoiser->setProgressCallback([this](float i_progress) { m_uDenoisingProgressBar->setValue(i_progress); drawAll(); });

	{
		Chronometer chrono;
		chrono.start();
		uDenoiser->denoise();
		chrono.stop();
		cout << "Denoising was performed in " << chrono << endl;
	}


//				checkAndPutToZeroNegativeInfNaNValues(outputDenoisedColorImage); // TODO: put in utils?

	m_uDenoisingProgressBar->setValue(1.f);

	setTextureData(ETexture::colorOutput, *m_uOutputImage);

	m_currentDisplayType = EDisplayType::colorOutput;

	bool skipOutputSave = false;
	if(m_outputFilePath.m_filePath == "")
		skipOutputSave = true;
	else if(!ifstream(m_outputFilePath.m_filePath))
		if(!ofstream(m_outputFilePath.m_filePath))
		{
			cerr << "Error: invalid output path '" << m_outputFilePath.m_filePath << "'!" << endl;
			skipOutputSave = true;
		}

	if(!skipOutputSave)
	{
		ImageIO::writeEXR(*m_uOutputImage, m_outputFilePath.m_filePath.c_str());
		cout << "Saved denoised output in file " << m_outputFilePath.m_filePath.c_str() << endl;
	}
	else
	{
		cout << "Warning: the denoised output has not been saved yet!" << endl;
	}

}


void GuiWindow::buildParametersSubWindow()
{
	m_uParametersSubWindow.reset(m_uFormHelper->addWindow(Eigen::Vector2i(10, 10), "BCD parameters"));

	m_uFormHelper->addVariable("Load/Save input files", m_pipelineParametersSelector.m_inputFileNames);
	m_uFormHelper->addVariable("Load/Save prefiltering params", m_pipelineParametersSelector.m_prefilteringParameters);
	m_uFormHelper->addVariable("Load/Save algo parameters", m_pipelineParametersSelector.m_denoiserParameters);
	m_uFormHelper->addButton("Load...", [this]() { loadInputsAndParameters(); });
	m_uFormHelper->addButton("Save...", [this]() { saveInputsAndParameters(); });

	m_uFormHelper->addGroup("Input files");
	auto inputColorWidget = m_uFormHelper->addVariable("Color image", m_colorInputFilePath);
	inputColorWidget->setCallback([this](const FilePathFormVariable& i_rFilePath) { loadInputColorFile(i_rFilePath.m_filePath); });
	auto inputHistWidget = m_uFormHelper->addVariable("Histogram image", m_histoInputFilePath);
	inputHistWidget->setCallback([this](const FilePathFormVariable& i_rFilePath) { loadInputHistoFile(i_rFilePath.m_filePath); });
	auto inputCovWidget = m_uFormHelper->addVariable("Covariance image", m_covarInputFilePath);
	inputCovWidget->setCallback([this](const FilePathFormVariable& i_rFilePath) { loadInputCovarFile(i_rFilePath.m_filePath); });

	m_uFormHelper->addGroup("Spike-removal prefiltering parameters");
	m_uFormHelper->addVariable("Perform spike removal", m_pipelineParameters.m_prefilteringParameters.m_performSpikeRemoval);
	m_uFormHelper->addVariable("Threshold (st. dev. factor)", m_pipelineParameters.m_prefilteringParameters.m_spikeRemovalThresholdStDevFactor);
	m_uFormHelper->addButton("Apply spike removal", [this]() { prefilter(); });

	m_uFormHelper->addGroup("Algo parameters");
	m_uFormHelper->addVariable("Nb of scales", m_pipelineParameters.m_denoiserParameters.m_nbOfScales);
	DenoiserParameters& dp = m_pipelineParameters.m_denoiserParameters.m_monoscaleParameters;
	m_uFormHelper->addVariable("Hist distance threshold", dp.m_histogramDistanceThreshold);
	m_uFormHelper->addVariable("Use CUDA", dp.m_useCuda);
	m_uFormHelper->addVariable("Nb of cores used (0 = default)", dp.m_nbOfCores);
	m_uFormHelper->addVariable("Patch radius", dp.m_patchRadius);
	m_uFormHelper->addVariable("Search window radius", dp.m_searchWindowRadius);
	m_uFormHelper->addVariable("Random pixel order", dp.m_useRandomPixelOrder);
	m_uFormHelper->addVariable("Marked pixels skipping probability", dp.m_markedPixelsSkippingProbability);
	m_uFormHelper->addVariable("Min eigen value", dp.m_minEigenValue);

	m_uFormHelper->addGroup("Outputs");
	auto outputFileWidget = m_uFormHelper->addVariable("Output file", m_outputFilePath);
	m_uFormHelper->addButton("Save as...",
			[this]()
			{
				string filePath = file_dialog({ { "exr", "EXR hdr image" } }, true);
				if(filePath == "")
					return;
				ImageIO::writeEXR(*m_uOutputImage, filePath.c_str());
				cout << "Saved denoised output in file " << filePath << endl;
			});

	auto denoiseButton = m_uFormHelper->addButton("Denoise", [this]() { denoise(); });
//	denoiseButton->setEnabled(false);

	m_uDenoisingProgressBar.reset(new ProgressBar(m_uParametersSubWindow.get()));
	m_uDenoisingProgressBar->setValue(0.f);
	m_uFormHelper->addWidget("", m_uDenoisingProgressBar.get());


//	performLayout();
	m_uParametersSubWindow->center();


	// begin of tmp adds to ease testing
//	inputColorWidget->setValue(FilePathFormVariable("/your/absolute/path/to/bcd/data/inputs/test.exr"));
//	inputColorWidget->callback();
//	inputHistWidget->setValue(FilePathFormVariable("/your/absolute/path/to/bcd/data/inputs/test_hist.exr"));
//	inputHistWidget->callback();
//	inputCovWidget->setValue(FilePathFormVariable("/your/absolute/path/to/bcd/data/inputs/test_cov.exr"));
//	inputCovWidget->callback();
//	outputFileWidget->setValue(FilePathFormVariable("/your/absolute/path/to/bcd/data/outputs/tmp/test_BCDfiltered.exr"));
//	outputFileWidget->callback();
//	loadInputsAndParameters("/your/absolute/path/to/bcd/data/inputs/json/test.bcd.json");
	// end of tmp adds to ease testing

}

void GuiWindow::buildDisplaySubWindow()
{
	m_uDisplaySubWindow.reset(m_uFormHelper->addWindow(Eigen::Vector2i(10, 10), "BCD display"));

	m_uFormHelper->addGroup("Current display");
	m_uFormHelper->addButton("Previous", [this](){ previousDisplayType(); });
	m_uFormHelper->addVariable("Display", m_currentDisplayType)
			->setItems({ "color input", "variance input", "pref. color input", "pref. variance input", "color output" });
	m_uFormHelper->addButton("Next", [this](){ nextDisplayType(); });

	m_uFormHelper->addGroup("Visibility of displays");
	m_uFormHelper->addVariable("Input color", m_displayTypeIsVisible[size_t(EDisplayType::colorInput)]);
	m_uFormHelper->addVariable("Input covariance", m_displayTypeIsVisible[size_t(EDisplayType::covTraceInput)]);
	m_uFormHelper->addVariable("Prefiltered input color", m_displayTypeIsVisible[size_t(EDisplayType::prefilteredColorInput)]);
	m_uFormHelper->addVariable("Prefiltered input covariance", m_displayTypeIsVisible[size_t(EDisplayType::prefilteredCovTraceInput)]);
	m_uFormHelper->addVariable("Output", m_displayTypeIsVisible[size_t(EDisplayType::colorOutput)]);

	m_uFormHelper->addGroup("Tonemapping parameters");
	m_uFormHelper->addVariable("Gamma", m_gamma);
	m_uFormHelper->addVariable("Exposure", m_exposure);
	m_uFormHelper->addVariable("Covariance scale", m_covTraceScale);

	m_uFormHelper->addGroup("Helix color map parameters");
	m_uFormHelper->addVariable("Max value", m_helix.m_maxValue);
	m_uFormHelper->addVariable("Start", m_helix.m_start);
	m_uFormHelper->addVariable("Rotations", m_helix.m_rotations);
	m_uFormHelper->addVariable("Hue", m_helix.m_hue);
	m_uFormHelper->addVariable("Gamma", m_helix.m_gamma);

}

void GuiWindow::buildGui()
{
	m_uFormHelper.reset(new FormHelper(this));

	buildParametersSubWindow();
	buildDisplaySubWindow();

	performLayout();

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

//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0,
//				GL_RGB, GL_FLOAT, defaultData.data());
	}
}

void GuiWindow::initShaders()
{
	string vsWithoutUVs = R"(
#version 330
uniform mat4 modelViewProj;
in vec3 position;
void main()
{
	gl_Position = modelViewProj * vec4(position, 1.0);
}
			)";

	string vsWithUVs = R"(
#version 330
uniform mat4 modelViewProj;
in vec3 position;
in vec2 vertexTexCoords;
out vec2 texCoords;
void main()
{
	gl_Position = modelViewProj * vec4(position, 1.0);
	texCoords = vertexTexCoords;
}
			)";

	string fsEmpty = R"(
#version 330
out vec4 color;
void main()
{
	color = vec4(0.0, 0.0, 0.0, 1.0);
}
			)";

	string fsColor = R"(
#version 330
in vec2 texCoords;
out vec4 color;
uniform sampler2D textureSampler;
void main()
{
	color = vec4(texture(textureSampler, texCoords).rgb, 1.0);
//	color = vec4(texCoords.x, 0, texCoords.y, 1.0);\n"
}
			)";

	string fsColorTonemapped = R"(
#version 330
in vec2 texCoords;
out vec4 color;
uniform sampler2D textureSampler;
uniform float gamma = 2.2;
uniform float exposure = 1.0;
void main()
{
	color = vec4(exposure * pow(texture(textureSampler, texCoords).rgb, vec3(1.0, 1.0, 1.0) / gamma), 1.0);
}
			)";

	string fsScalar = R"(
#version 330
in vec2 texCoords;
out vec4 color;
uniform sampler2D textureSampler;
void main()
{
	float scalar = texture(textureSampler, texCoords).r;
	color = vec4(scalar, scalar, scalar, 1.0);
}
			)";

	string fsScalarTonemapped = R"(
#version 330
in vec2 texCoords;
out vec4 color;
uniform sampler2D textureSampler;
uniform float gamma = 2.2;
uniform float exposure = 1.0;
void main()
{
	float scalar = exposure * pow(texture(textureSampler, texCoords).r, 1.0 / gamma);
	color = vec4(scalar, scalar, scalar, 1.0);
}
			)";

	string fsScalarHelix = R"(
#version 330
in vec2 texCoords;
out vec4 color;
uniform sampler2D textureSampler;
uniform float maxValue = 1.0;
uniform float start = 0.5;
uniform float rotations = -1.5;
uniform float hue = 1.0;
uniform float gamma = 2.2;
// see http://astron-soc.in/bulletin/11June/289392011.pdf
void main()
{
	float scalar = min(1.0, texture(textureSampler, texCoords).r / maxValue);
	float scalarGamma = pow(scalar, 1.0 / gamma);
	float angle = 2.0 * 3.14159265359 * (start / 3.0 + rotations * scalar);
	float cosangle = cos(angle);
	float sinangle = sin(angle);
	float amplitude = hue * scalarGamma * (1.0 - scalarGamma) * 0.5;
	color = vec4(
			scalarGamma * vec3(1.0, 1.0, 1.0) +
			amplitude * vec3(
					-0.14861 * cosangle + 1.78277 * sinangle,
					-0.29227 * cosangle - 0.90649 * sinangle,
					1.97294 * cosangle
			),
			1.0);
}
			)";

	m_shaderPrograms[size_t(EShaderProgram::empty)].init("empty", vsWithoutUVs, fsEmpty);
	m_shaderPrograms[size_t(EShaderProgram::colorImage)].init("color image", vsWithUVs, fsColor);
	m_shaderPrograms[size_t(EShaderProgram::colorImageTonemapped)].init("color image with tone mapping", vsWithUVs, fsColorTonemapped);
	m_shaderPrograms[size_t(EShaderProgram::scalarImage)].init("scalar image", vsWithUVs, fsScalar);
	m_shaderPrograms[size_t(EShaderProgram::scalarImageTonemapped)].init("scalar image with tone mapping", vsWithUVs, fsScalarTonemapped);
	m_shaderPrograms[size_t(EShaderProgram::scalarImageHelix)].init("scalar image with helix color map", vsWithUVs, fsScalarHelix);

}

bool GuiWindow::shaderNeedsTexture(EShaderProgram i_shaderProgram)
{
	assert(i_shaderProgram != EShaderProgram::count);
	switch(i_shaderProgram)
	{
	case EShaderProgram::empty:
		return false;
	default:
		return true;
	}
}

void GuiWindow::initOpenGL()
{


	// glEnable(GL_TEXTURE_2D);

	initTextures();
	initShaders();

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

	for(size_t i = 0; i < m_shaderPrograms.size(); ++i)
	{
		GLShader& rShaderProgram = m_shaderPrograms[i];
		rShaderProgram.bind();
		rShaderProgram.uploadIndices(indices);
		rShaderProgram.uploadAttrib("position", positions);
		if(shaderNeedsTexture(EShaderProgram(i)))
		{
			rShaderProgram.uploadAttrib("vertexTexCoords", vertexTexCoords);
			rShaderProgram.setUniform("textureSampler", GLuint(0));
		}
	}

}

GuiWindow::EShaderProgram GuiWindow::getShaderProgramFromDisplayType(EDisplayType i_displayType)
{
	switch (i_displayType)
	{
	case EDisplayType::colorInput:
	case EDisplayType::prefilteredColorInput:
	case EDisplayType::colorOutput:
		return EShaderProgram::colorImageTonemapped;
	case EDisplayType::covTraceInput:
	case EDisplayType::prefilteredCovTraceInput:
		return EShaderProgram::scalarImageHelix; // or scalarImageTonemapped
	default:
		assert(false);
		return EShaderProgram::empty;
	}
}

void GuiWindow::setCurrentShaderProgram()
{
	if(!isLoaded(m_currentDisplayType))
		m_currentShaderProgramType = EShaderProgram::empty;
	else
		m_currentShaderProgramType = getShaderProgramFromDisplayType(m_currentDisplayType);
	m_pCurrentShaderProgram = &m_shaderPrograms[size_t(m_currentShaderProgramType)];
}

GuiWindow::ETexture GuiWindow::getTextureFromDisplayType(EDisplayType i_displayType)
{
	switch (i_displayType)
	{
	case EDisplayType::colorInput: return ETexture::colorInput;
	case EDisplayType::covTraceInput: return ETexture::covTraceInput;
	case EDisplayType::prefilteredColorInput: return ETexture::prefilteredColorInput;
	case EDisplayType::prefilteredCovTraceInput: return ETexture::prefilteredCovTraceInput;
	case EDisplayType::colorOutput: return ETexture::colorOutput;
	default:
		assert(false);
		return ETexture::colorInput;
	}
}

void GuiWindow::onDisplayTypeChange()
{
	m_uFormHelper->refresh();

	setCurrentShaderProgram();

	const DeepImage<float>* pCurrentImage = nullptr;
	switch(m_currentDisplayType)
	{
	case EDisplayType::colorInput:
		pCurrentImage = m_uColorInputImage.get();
		break;
	case EDisplayType::covTraceInput:
		pCurrentImage = m_uCovTraceInputImage.get();
		break;
	case EDisplayType::prefilteredColorInput:
		pCurrentImage = m_uPrefilteredColorInputImage.get();
		break;
	case EDisplayType::prefilteredCovTraceInput:
		pCurrentImage = m_uPrefilteredCovTraceInputImage.get();
		break;
	case EDisplayType::colorOutput:
		pCurrentImage = m_uOutputImage.get();
		break;
	default:
		assert(false);
		break;
	}

	if(!pCurrentImage->isEmpty())
	{
		m_viewFrame.changeResolution(width(), height(), pCurrentImage->getWidth(), pCurrentImage->getHeight());
		m_viewChanged = true;
	}

}

void GuiWindow::drawContents()
{
	if(m_currentDisplayType != m_oldDisplayType)
		m_displayChanged = true;

	if(m_displayChanged) // m_displayChanged can also be set to true when an image is loaded for example
		onDisplayTypeChange();

	m_pCurrentShaderProgram->bind();

	if(shaderNeedsTexture(m_currentShaderProgramType))
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_textureIds[size_t(getTextureFromDisplayType(m_currentDisplayType))]);
	}

	if(m_viewChanged)
		setCamera();

	switch(m_currentShaderProgramType)
	{
	case EShaderProgram::colorImageTonemapped:
		m_pCurrentShaderProgram->setUniform("gamma", m_gamma);
		m_pCurrentShaderProgram->setUniform("exposure", m_exposure);
		break;
	case EShaderProgram::scalarImageTonemapped:
		m_pCurrentShaderProgram->setUniform("gamma", m_gamma);
		m_pCurrentShaderProgram->setUniform("exposure", m_exposure * m_covTraceScale);
		break;
	case EShaderProgram::scalarImageHelix:
		m_pCurrentShaderProgram->setUniform("maxValue", m_helix.m_maxValue);
		m_pCurrentShaderProgram->setUniform("start", m_helix.m_start);
		m_pCurrentShaderProgram->setUniform("rotations", m_helix.m_rotations);
		m_pCurrentShaderProgram->setUniform("hue", m_helix.m_hue);
		m_pCurrentShaderProgram->setUniform("gamma", m_helix.m_gamma);
		break;
	}

	if(m_currentShaderProgramType != EShaderProgram::empty)
		m_pCurrentShaderProgram->drawIndexed(GL_TRIANGLES, 0, 2);

	m_displayChanged = false;
	m_viewChanged = false;
}


void GuiWindow::setCamera()
{
//	m_viewFrame.print();

	Eigen::Matrix4f mvp;
	mvp.setIdentity();
	float zoomFactorX = 2.f / m_viewFrame.m_width;
	float zoomFactorY = 2.f / m_viewFrame.m_height;
	float dx = -1 - m_viewFrame.m_xMin * zoomFactorX;
	float dy = -1 - m_viewFrame.m_yMin * zoomFactorY;

	mvp(0,0) = zoomFactorX;
	mvp(1,1) = zoomFactorY;
	mvp(0,3) = dx;
	mvp(1,3) = dy;

	m_pCurrentShaderProgram->setUniform("modelViewProj", mvp);
}


bool GuiWindow::isLoaded(EDisplayType i_displayType)
{
	switch(i_displayType)
	{
	case EDisplayType::colorInput: return !m_uColorInputImage->isEmpty();
	case EDisplayType::covTraceInput: return !m_uCovInputImage->isEmpty();
	case EDisplayType::prefilteredColorInput: return !m_uPrefilteredColorInputImage->isEmpty();
	case EDisplayType::prefilteredCovTraceInput: return !m_uPrefilteredCovInputImage->isEmpty();
	case EDisplayType::colorOutput: return !m_uOutputImage->isEmpty();
	}
	return false;
}

void GuiWindow::previousDisplayType()
{
	size_t i0 = size_t(m_currentDisplayType);
	size_t i = i0;
	do i = (i == 0 ? size_t(EDisplayType::count) - 1 : i - 1); while(i != i0 && !m_displayTypeIsVisible[i]);
	m_currentDisplayType = static_cast<EDisplayType>(i);
}

void GuiWindow::nextDisplayType()
{
	size_t i0 = size_t(m_currentDisplayType);
	size_t i = i0;
	do i = (i == size_t(EDisplayType::count) - 1 ? 0 : i + 1); while(i != i0 && !m_displayTypeIsVisible[i]);
	m_currentDisplayType = static_cast<EDisplayType>(i);
}


void ViewFrame::reset(int i_windowWidth, int i_windowHeight, int i_imageWidth, int i_imageHeight)
{
	m_initialWidth = 2.0f * float(i_windowWidth) / float(i_imageWidth);
	m_initialHeight = 2.0f * float(i_windowHeight) / float(i_imageHeight);
	m_width = m_initialWidth;
	m_height = m_initialHeight;
	m_xMin = -0.5f * m_initialWidth;
	m_yMin = -0.5f * m_initialHeight;
	m_totalZoomExponent = 0.f;
}

void ViewFrame::resetZoomAndRecenter()
{
	m_width = m_initialWidth;
	m_height = m_initialHeight;
	m_xMin = -0.5f * m_width;
	m_yMin = -0.5f * m_height;
	m_totalZoomExponent = 0.f;
}

void ViewFrame::changeResolution(int i_windowWidth, int i_windowHeight, int i_imageWidth, int i_imageHeight)
{
	float xCenter = m_xMin + 0.5f * m_width;
	float yCenter = m_yMin + 0.5f * m_height;

	m_initialWidth = 2.0f * float(i_windowWidth) / float(i_imageWidth);
	m_initialHeight = 2.0f * float(i_windowHeight) / float(i_imageHeight);
	m_width = m_initialWidth * pow(s_zoomFactor, m_totalZoomExponent);
	m_height = m_initialHeight * pow(s_zoomFactor, m_totalZoomExponent);
	m_xMin = xCenter - 0.5f * m_width;
	m_yMin = yCenter - 0.5f * m_height;
}

void ViewFrame::zoom(int i_windowWidth, int i_windowHeight, int i_targetPixelX, int i_targetPixelY, float i_zoomWheelIncrease)
{
	float x = m_xMin + (m_width * i_targetPixelX) / i_windowWidth;
	float y = m_yMin + (m_height * (i_windowHeight - i_targetPixelY)) / i_windowHeight;
	m_totalZoomExponent -= i_zoomWheelIncrease * s_wheelFactor;
	m_width = m_initialWidth * pow(s_zoomFactor, m_totalZoomExponent);
	m_height = m_initialHeight * pow(s_zoomFactor, m_totalZoomExponent);
	m_xMin = x - (m_width * i_targetPixelX) / i_windowWidth;
	m_yMin = y - (m_height * (i_windowHeight - i_targetPixelY)) / i_windowHeight;
}

void ViewFrame::print()
{
	cout << "viewport (x,y) in [" << m_xMin << ", " << m_xMin + m_width << "] x [" << m_yMin << ", " << m_yMin + m_height << "]" << endl;
}


bool GuiWindow::mouseButtonEvent(const Vector2i &p, int button, bool down, int modifiers)
{
//	cout << "Entering GuiWindow::mouseButtonEvent(p = " << p << ", button = " << button << ", down = " << down << ", modifiers = " << modifiers << ")" << endl;
	if(Widget::mouseButtonEvent(p, button, down, modifiers))
		return true;
//	cout << "passed super call Widget::mouseButtonEvent" << endl;

	if(down)
		m_mouseMovedBeforeButtonRelease = false;
	else if(!m_mouseMovedBeforeButtonRelease)
		mouseClickEvent(p, button, modifiers);

	return true;
}

bool GuiWindow::mouseMotionEvent(const Vector2i &p, const Vector2i &rel, int buttons, int modifiers)
{
	// cout << "Entering GuiWindow::mouseMotionEvent(p = " << p << ", rel = " << rel << ", button = " << button << ", modifiers = " << modifiers << ")" << endl;
	if(Widget::mouseMotionEvent(p, rel, buttons, modifiers))
		return true;
	// cout << "passed super call Widget::mouseMotionEvent" << endl;

	if(buttons)
		m_mouseMovedBeforeButtonRelease = true;

	if(buttons & ((1 << GLFW_MOUSE_BUTTON_2) | (1 << GLFW_MOUSE_BUTTON_3)))
	{
		m_viewFrame.m_xMin -= (m_viewFrame.m_width * float(rel(0))) / float(width());
		m_viewFrame.m_yMin += (m_viewFrame.m_height * float(rel(1))) / float(height());
		m_viewChanged = true;
	}

	return true;

}


bool GuiWindow::mouseDragEvent(const Vector2i &p, const Vector2i &rel, int button, int modifiers)
{
//	cout << "Entering GuiWindow::mouseDragEvent(p = " << p << ", rel = " << rel << ", button = " << button << ", modifiers = " << modifiers << ")" << endl;
	if(Widget::mouseDragEvent(p, rel, button, modifiers))
		return true;
//	cout << "passed super call Widget::mouseDragEvent" << endl;



	return true;

}

bool GuiWindow::scrollEvent(const Vector2i &p, const Vector2f &rel)
{
//	cout << "Entering GuiWindow::scrollEvent(p = " << p << ", rel = " << rel << ")" << endl;
	if(Widget::scrollEvent(p, rel))
		return true;
//	cout << "passed super call Widget::scrollEvent" << endl;

	m_viewFrame.zoom(width(), height(), p(0), p(1), rel(1));
	m_viewChanged = true;

//	update();


	return true;
}


bool GuiWindow::keyboardEvent(int key, int scancode, int action, int modifiers)
{
//	cout << "Entering GuiWindow::keyboardEvent(key = " << key << ", scancode = " << scancode << ", action = " << action << ", modifiers = " << modifiers << ")" << endl;
	if(Screen::keyboardEvent(key, scancode, action, modifiers))
		return true;
//	cout << "passed super call Widget::keyboardEvent" << endl;

	if(action != GLFW_PRESS)
		return false;

	switch(key)
	{
	case GLFW_KEY_SPACE:
		m_viewFrame.resetZoomAndRecenter();
		m_viewChanged = true;
		return true;
	case GLFW_KEY_ESCAPE:
		setVisible(false);
		return true;
	case GLFW_KEY_UP:
		previousDisplayType();
		return true;
	case GLFW_KEY_DOWN:
		nextDisplayType();
		return true;
	case GLFW_KEY_TAB:
//		cout << "Tab pressed!" << endl;
		m_uParametersSubWindow->setVisible(m_hideAllSubWindows);
		m_uDisplaySubWindow->setVisible(m_hideAllSubWindows);
//		cout << "m_uParametersSubWindow is " << (m_uParametersSubWindow->visible() ? "" : "not ") << "visible" << endl;
		m_hideAllSubWindows = !m_hideAllSubWindows;
//		cout << "m_hideAllSubWindows = " << m_hideAllSubWindows << endl;
		return true;
	}

	return false;
}

bool GuiWindow::mouseClickEvent(const Eigen::Vector2i &p, int button, int modifiers)
{
//	cout << "Entering GuiWindow::mouseClickEvent(p = " << p << ", button = " << button << ", modifiers = " << modifiers << ")" << endl;

	switch(button)
	{
	case GLFW_MOUSE_BUTTON_2:
//		m_displayView.m_xMin = -0.5f * m_displayView.m_width;
//		m_displayView.m_yMin = -0.5f * m_displayView.m_height;
//		setCamera();
		break;
	case GLFW_MOUSE_BUTTON_3:
		{
//			int w = width();
//			int h = height();
//			int mouseX = p(0);
//			int mouseY = p(1);
//			float x = m_displayView.m_xMin + (m_displayView.m_width * mouseX) / w;
//			float y = m_displayView.m_yMin + (m_displayView.m_height * (h - mouseY)) / h;
//			m_displayView.m_totalZoomExponent = 0;
//			m_displayView.m_width = m_displayView.m_initialWidth;
//			m_displayView.m_height = m_displayView.m_initialHeight;
//			m_displayView.m_xMin = x - (m_displayView.m_width * mouseX) / w;
//			m_displayView.m_yMin = y - (m_displayView.m_height * (h - mouseY)) / h;
//			setCamera();
			break;
		}
	}


	return true;
}
