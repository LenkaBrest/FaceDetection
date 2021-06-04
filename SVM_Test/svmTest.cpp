// -svm=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\SVM_Train\SVM_MARC.yaml -pos=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\positive -neg=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\negative

#include "svmTest.h"

using namespace cv;

int main(int argc, const char** argv)
{
#pragma region Argument parsing

	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{svm |   | path to the trained svm}"
						  "{pos p positive  |   | path for the positiv testimages}"
						  "{neg n negative  |   | path for the negativ testimages}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("FaceDetection-HOG-SVM");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String svmPath = parser.get<String>("svm");
	String positivePath = parser.get<String>("pos");
	String negativePath = parser.get<String>("neg");

	if (positivePath == "")
	{
		printf("There is no positivePath\n");
		return -1;
	}
	if (negativePath == "")
	{
		printf("There is no negativePath\n");
		return -1;
	}

#pragma endregion

	bool test = testSVM(&positivePath, &negativePath, &svmPath);
	if (!test)
		return -1;
	return 0;
}


bool testSVM(String* positiveTestPath, String* negativTestPath, String* svmPath)
{
#pragma region Initialization

	printf("Initialize\n");
	// Finding all images in both pathes
	std::vector<String> positiveFileNames, negativeFileNames, allFileNames;
	glob(*positiveTestPath, positiveFileNames);
	glob(*negativTestPath, negativeFileNames);

	// Testing if there are images in the pathes
	if (positiveFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", *positiveTestPath);
		return false;
	}
	if (negativeFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", *negativTestPath);
		return false;
	}
	allFileNames.insert(allFileNames.end(), positiveFileNames.begin(), positiveFileNames.end());
	allFileNames.insert(allFileNames.end(), negativeFileNames.begin(), negativeFileNames.end());

	//Mat testData = Mat_<float>(DESCRIPTOR_SIZE, allFileNames.size());
	Mat testData = Mat_<float>(allFileNames.size(), DESCRIPTOR_SIZE);
	int testCount = 0;

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm = svm->load(*svmPath);
	if (!svm->isTrained())
	{
		printf("The SVM isn't trained through this path: %s\n", *svmPath);
		return false;
	}

	HOGDescriptor hogD;
	hogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);
	std::vector<float> descriptorsValues;
	std::vector<Point> locations;

	clock_t beginTime = clock();

#pragma endregion

#pragma region HOG Descriptors

	// Converting the positve images and calculating the HOG
	std::cout << "Calculate all Images (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	for (std::vector<String>::iterator fileName = allFileNames.begin(); fileName != allFileNames.end(); ++fileName)
	{

		Mat actualImage = imread(*fileName);

		// Testing if the file is an image
		if (actualImage.empty())
		{
			printf("Couldn't read the image %s\n", *fileName);
			return false;
		}
		cvtColor(actualImage, actualImage, cv::COLOR_BGR2GRAY);
		resize(actualImage, actualImage, Size(WINDOW_SIZE, WINDOW_SIZE));

		// Calculating the HOG
		hogD.compute(actualImage, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		// I need to transpose to get every sample in a column and not in a row
		Mat descriptorsVector = Mat_<float>(descriptorsValues, true);
		transpose(descriptorsVector, descriptorsVector);
		descriptorsVector.row(0).copyTo(testData.row(testCount));
		testCount++;

	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;
	std::cout << std::endl << std::endl;

#pragma endregion

#pragma region Testing the Data

	Mat results;
	svm->predict(testData, results);
	int fPos = 0, fNeg = 0;

	for (int c = 0; c < allFileNames.size(); c++)
	{
		float result = results.at<float>(c, 0);
		if (c < positiveFileNames.size() && result != 1)
			fNeg++;
		else if (c >= positiveFileNames.size() && result != -1)
			fPos++;
	}
	printf("False Positive: %d\n", fPos);
	printf("False Negative: %d\n", fNeg);

#pragma endregion

	return true;
}
