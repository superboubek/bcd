
#ifndef GUI_WINDOW_H
#define GUI_WINDOW_H

#include "IDenoiser.h"

#include "nanogui/glutil.h"
#include "nanogui/screen.h"

#include <string>

#include <vector>
#include <array>

#include <memory>
#include <cstdint>


namespace nanogui
{
	class FormHelper;
}

namespace bcd
{

	template<class T> class DeepImage;

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

		void resetZoomAndRecenter();

		void print();

	};

	class GuiWindow : public nanogui::Screen
	{
	public:
		enum class EDisplayType
		{
			colorInput,
			covTraceInput,
			colorOutput
		};
		static const std::size_t nbOfDisplayTypes = 3;

		enum class ETexture
		{
			colorInput,
			covTraceInput,
			colorOutput,
			count
		};

		enum class EShaderProgram
		{
			empty,
			colorImage,
			colorImageTonemapped,
			scalarImage,
			count
		};


	public:

		GuiWindow();
		~GuiWindow();

		void displayUntilClosed();

		virtual void drawContents();


//		bool cursorPosCallbackEvent(double x, double y);
//		virtual bool mouseButtonEvent(int button, int action, int modifiers);
//		bool keyCallbackEvent(int key, int scancode, int action, int mods);
//		bool charCallbackEvent(unsigned int codepoint);
//		bool dropCallbackEvent(int count, const char **filenames);
//		virtual bool scrollEvent(double x, double y);
//		bool resizeCallbackEvent(int width, int height);

		virtual bool mouseButtonEvent(const Eigen::Vector2i &p, int button, bool down, int modifiers) override;
		virtual bool mouseMotionEvent(const Eigen::Vector2i &p, const Eigen::Vector2i &rel, int button, int modifiers) override;
		virtual bool mouseDragEvent(const Eigen::Vector2i &p, const Eigen::Vector2i &rel, int button, int modifiers) override;
		virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &rel) override;
		virtual bool keyboardEvent(int key, int scancode, int action, int modifiers) override;

		bool mouseClickEvent(const Eigen::Vector2i &p, int button, int modifiers);

		bool m_mouseMovedBeforeButtonRelease;

	private:
		void buildParametersSubWindow();
		void buildDisplaySubWindow();
		void buildGui();
		void initTextures();
		void initOpenGL();
		void setCamera();

		bool isLoaded(EDisplayType i_displayType);

		void previousDisplayType();
		void nextDisplayType();
		void onDisplayTypeChange();

	private:

		std::unique_ptr<nanogui::FormHelper> m_uFormHelper;

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
		std::array<nanogui::GLShader, std::size_t(EShaderProgram::count)> m_shaderPrograms;

		std::array<GLuint, std::size_t(ETexture::count)> m_textureIds;

		DisplayView m_displayView;

		EDisplayType m_currentDisplayType;
		EDisplayType m_lastDisplayType;
		std::array<bool, nbOfDisplayTypes> m_displayTypeIsVisible;


	};

}
#endif // GUI_WINDOW_H
