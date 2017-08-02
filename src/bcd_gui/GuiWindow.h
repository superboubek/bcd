
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



	class GuiWindow : public nanogui::Screen
	{
	public:
		GuiWindow();
		~GuiWindow();

		void displayUntilClosed();

		virtual void drawContents();


	private:
		void buildGui();
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


	};

}
#endif // GUI_WINDOW_H
