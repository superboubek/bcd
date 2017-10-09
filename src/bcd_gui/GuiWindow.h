
#ifndef GUI_WINDOW_H
#define GUI_WINDOW_H

#include "IDenoiser.h"

#include "ParametersIO.h"

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
	class Window;
	class ProgressBar;
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


	/// @brief Position and size of the viewing frame in view space
	struct ViewFrame
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

		ViewFrame() :
				m_initialWidth(2.f),
				m_initialHeight(2.f),
				m_width(2.f),
				m_height(2.f),
				m_xMin(-1.f),
				m_yMin(-1.f),
				m_totalZoomExponent(0.f)
		{}

		void reset(int i_windowWidth, int i_windowHeight, int i_imageWidth, int i_imageHeight);
		void resetZoomAndRecenter();
		void changeResolution(int i_windowWidth, int i_windowHeight, int i_imageWidth, int i_imageHeight);
		void zoom(int i_windowWidth, int i_windowHeight, int i_targetPixelX, int i_targetPixelY, float i_zoomWheelIncrease);

		void print();

	};

	class GuiWindow : public nanogui::Screen
	{
	public:
		enum class EDisplayType
		{
			colorInput,
			covTraceInput,
			prefilteredColorInput,
			prefilteredCovTraceInput,
			colorOutput,
			count
		};

		enum class ETexture
		{
			colorInput,
			covTraceInput,
			prefilteredColorInput,
			prefilteredCovTraceInput,
			colorOutput,
			count
		};

		enum class EShaderProgram
		{
			empty,
			colorImage,
			colorImageTonemapped,
			scalarImage,
			scalarImageTonemapped,
			scalarImageHelix,
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
		void setTextureData(ETexture i_texture, const DeepImage<float>& i_rImage);

		void loadInputsAndParameters();
		void loadInputsAndParameters(const std::string& i_rFilePath);
		void saveInputsAndParameters();

		void loadInputColorFile(const std::string& i_rFilePath);
		void loadInputHistoFile(const std::string& i_rFilePath);
		void loadInputCovarFile(const std::string& i_rFilePath);

		static void computeCovTraceImage(DeepImage<float>& o_rCovTraceImage, const DeepImage<float>& i_rCovImage);

		void prefilter();
		void denoise();

		void buildParametersSubWindow();
		void buildDisplaySubWindow();
		void buildGui();
		void initTextures();
		void initShaders();

		bool shaderNeedsTexture(EShaderProgram i_shaderProgram);

		void initOpenGL();
		EShaderProgram getShaderProgramFromDisplayType(EDisplayType i_displayType);
		void setCurrentShaderProgram();
		ETexture getTextureFromDisplayType(EDisplayType i_displayType);

		void setCamera();

		bool isLoaded(EDisplayType i_displayType);

		void previousDisplayType();
		void nextDisplayType();
		void onDisplayTypeChange();

	private:

		std::unique_ptr<nanogui::FormHelper> m_uFormHelper;

		std::unique_ptr<nanogui::Window> m_uParametersSubWindow;
		std::unique_ptr<nanogui::Window> m_uDisplaySubWindow;

		std::unique_ptr<nanogui::ProgressBar> m_uDenoisingProgressBar;

		bool m_hideAllSubWindows;

		FilePathFormVariable m_colorInputFilePath;
		FilePathFormVariable m_histoInputFilePath;
		FilePathFormVariable m_covarInputFilePath;

		bool m_inputsAreLoaded;

		FilePathFormVariable m_outputFilePath;

		PipelineParameters m_pipelineParameters;
		PipelineParametersSelector m_pipelineParametersSelector;

		DenoiserInputs m_denoiserInputs;
		DenoiserOutputs m_denoiserOutputs;

		std::unique_ptr< DeepImage<float> > m_uColorInputImage;
		std::unique_ptr< DeepImage<float> > m_uNbOfSamplesInputImage;
		std::unique_ptr< DeepImage<float> > m_uHistInputImage;
		std::unique_ptr< DeepImage<float> > m_uCovInputImage;
		std::unique_ptr< DeepImage<float> > m_uCovTraceInputImage;

		std::unique_ptr< DeepImage<float> > m_uPrefilteredColorInputImage;
		std::unique_ptr< DeepImage<float> > m_uPrefilteredNbOfSamplesInputImage;
		std::unique_ptr< DeepImage<float> > m_uPrefilteredHistInputImage;
		std::unique_ptr< DeepImage<float> > m_uPrefilteredCovInputImage;
		std::unique_ptr< DeepImage<float> > m_uPrefilteredCovTraceInputImage;

		std::unique_ptr< DeepImage<float> > m_uOutputImage;


		std::array<nanogui::GLShader, std::size_t(EShaderProgram::count)> m_shaderPrograms;
		EShaderProgram m_currentShaderProgramType;
		nanogui::GLShader* m_pCurrentShaderProgram;

		std::array<GLuint, std::size_t(ETexture::count)> m_textureIds;

		ViewFrame m_viewFrame;

		EDisplayType m_oldDisplayType;
		EDisplayType m_currentDisplayType;

		bool m_displayChanged;
		bool m_viewChanged;

		std::array<bool, size_t(EDisplayType::count)> m_displayTypeIsVisible;
		float m_gamma;
		float m_exposure;
		float m_covTraceScale;

		struct HelixColorMapParameters
		{
			float m_maxValue;
			float m_start;
			float m_rotations;
			float m_hue;
			float m_gamma;
			HelixColorMapParameters() :
					m_maxValue(1.f),
					m_start(0.5f),
					m_rotations(-1.5f),
					m_hue(1.f),
					m_gamma(2.2f) {}
		} m_helix;




	};

}
#endif // GUI_WINDOW_H
