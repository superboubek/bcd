
#ifndef GUI_WINDOW_H
#define GUI_WINDOW_H

#include "IDenoiser.h"

#include "nanogui/glutil.h"
#include "nanogui/screen.h"

#include <string>

#include <memory>


template<class T> class DeepImage;

namespace bcd
{

	struct FilePathFormVariable
	{
		std::string m_filePath;
		FilePathFormVariable() : m_filePath() {}
		FilePathFormVariable(const std::string& i_rFilePath) : m_filePath(i_rFilePath) {}
		bool operator==(const FilePathFormVariable& i_rOther) const
		{
			return (m_filePath == i_rOther.m_filePath);
		}
		bool operator!=(const FilePathFormVariable& i_rOther) const
		{
			return (m_filePath != i_rOther.m_filePath);
		}

	};


	struct DisplayView
	{
		float m_initialWidth;
		float m_initialHeight;
		float m_xMin;
		float m_yMin;
		float m_width;
		float m_height;
		float m_totalZoomExponent; // accumulation of wheel moves

		static const float s_zoomFactor;
		static const float s_wheelFactor;

		DisplayView() :
				m_initialWidth(2.f),
				m_initialHeight(2.f),
				m_width(2.f),
				m_height(2.f),
				m_xMin(-1.f),
				m_yMin(-1.f),
				m_totalZoomExponent(0.f)
		{}

		void reset(int windowWidth, int windowHeight, int imageWidth, int imageHeight);

		void print();

	};

	class GuiWindow : public nanogui::Screen
	{
	public:

		GuiWindow();
		~GuiWindow();

		void displayUntilClosed();

		virtual void drawContents();

		void setCamera();

//		bool cursorPosCallbackEvent(double x, double y);
//		virtual bool mouseButtonEvent(int button, int action, int modifiers);
//		bool keyCallbackEvent(int key, int scancode, int action, int mods);
//		bool charCallbackEvent(unsigned int codepoint);
//		bool dropCallbackEvent(int count, const char **filenames);
//		virtual bool scrollEvent(double x, double y);
//		bool resizeCallbackEvent(int width, int height);

		virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers);
		virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &rel);

	private:
		void buildGui();
		void initTextures();
		void initOpenGL();

	private:
		FilePathFormVariable m_colorInputFilePath;
		FilePathFormVariable m_histInputFilePath;
		FilePathFormVariable m_covInputFilePath;

		std::string m_loadedColorInputFilePath;
		std::string m_loadedHistInputFilePath;
		std::string m_loadedCovInputFilePath;

		bool m_inputsAreLoaded;

		FilePathFormVariable m_outputFilePath;

		int m_nbOfScales;
		DenoiserParameters m_denoiserParameters;
		DenoiserInputs m_denoiserInputs;
		DenoiserOutputs m_denoiserOutputs;

		std::unique_ptr< DeepImage<float> > m_uColorInputImage;
		std::unique_ptr< DeepImage<float> > m_uNbOfSamplesInputImage;
		std::unique_ptr< DeepImage<float> > m_uHistInputImage;
		std::unique_ptr< DeepImage<float> > m_uCovInputImage;
		std::unique_ptr< DeepImage<float> > m_uOutputImage;


		nanogui::GLShader m_shaderProgram;

		std::array<GLuint, 1> m_textureIds;

		DisplayView m_displayView;

	};

}
#endif // GUI_WINDOW_H
