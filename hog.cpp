#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/ml.hpp"
#include <ctype.h>
#include <math.h>
//#include <windows.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include<io.h>
#include <fstream>
#include <ctime>


#define HOW_LONG 0 /// wait until key or X is pressed, other value is in milisecs

using namespace std;
using namespace cv;
using namespace cv::ml;

vector< Mat > img_pos_lst, img_neg_list, test_lst;
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData );
Mat get_hogdescriptor_visual_image(Mat& origImg, vector<float>& descriptorValues, Size winSize, Size cellSize, int scaleFactor, double viz_factor);

int main(int argc, const char* argv[])
{
	string pathData;
	int posCount, negCount;
	cout << "Enter Positive Data directory path" << endl;
	cin>>pathData;
	
	srand(time(NULL));

	//Positive image count
	posCount = 5417;
	//Negative image count
	negCount = 5417;
	
	int in = 0;
	vector<int> test_gold;

	for(int i=1; i < posCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/"<< "pveimages" << "/" << i << ".jpg";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(),1);
		if (img.empty())
		{ 
			return -1;
		}
		
		resize(img, img, Size(64, 64));
		
		img_pos_lst.push_back(img.clone());
	}
	
	cout<<"Get out"<<endl;

	for (int i = 1; i < negCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/" << "nveimages" << "/" << "TrainNeg" <<"/"<<i<< ".jpg";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(), 1);
		if (img.empty())
		{
			return -1;
		}
		
		
		resize(img, img, Size(64, 64));
		
		img_neg_list.push_back(img.clone());
	}

	cout<<"Get out of TrainNeg"<<endl;
	
	
	HOGDescriptor hog;
	//cout<<"HOGDescriptor"<<endl;
	Mat gradMat;
	vector<int> labelsMat;
	Mat trainingDataMat;
	std::vector<float> descriptors;

	hog.winSize = Size(64, 64);
	hog.blockSize = Size(4, 4);
	hog.blockStride = Size(2, 2);
	hog.cellSize = Size(2, 2);

	
	//for negative data
	for (int i = 0; i < img_neg_list.size(); ++i)
	{
		hog.compute(img_neg_list[i], descriptors, Size(0, 0), Size(0, 0));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		
		labelsMat.push_back(1);
	}
	
	
	cout<<"neg_lst"<<endl;
	
	//For positive Data
	cout<<"before making pos feature"<<endl;
	for (int i = 0; i < img_pos_lst.size(); ++i)
	{
		hog.compute(img_pos_lst[i], descriptors, Size(0, 0), Size(0, 0));
		//cout<<descriptors.size()<<endl;	
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);		
		trainingDataMat.push_back(descMat);
		descriptors.clear();

		labelsMat.push_back(-1);
		//cout<<i<<endl;
	}
	cout<<"after making pos feature"<<endl;

	Ptr<SVM> svm = SVM::create();

        svm->setType(SVM::C_SVC);
        svm->setKernel( SVM::LINEAR );
        //svm->setGamma( 0 ); //3
        svm->setDegree( 1 );//3
    	svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    	
    	svm->setC( 0.01 ); // From paper, soft classifier
	//svm->setType(SVM::C_SVC);
	cout<<"set svm coef"<<endl;
	
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	
	svm->train(td);
	cout<<"trained"<<endl;
	
	
	Ptr<SVM> svmLoad;
	//svmLoad = SVM::load("hogSVMFaces.xml");
	//Mat loadSVMMat = svmLoad->getSupportVectors();
	vector<float> loadSVMvector;
	
	cout<<"load svm"<<endl;

	get_svm_detector(svm, loadSVMvector);
	HOGDescriptor hogTest;
	hogTest.winSize = Size(64, 64);
	hogTest.blockSize = Size(4, 4);
	hogTest.blockStride = Size(2, 2);
	hogTest.cellSize = Size(2, 2);
	//hogTest.nbins = 9;
	hogTest.setSVMDetector(loadSVMvector);
	hogTest.save("hogSVMFaces.xml");
	cout<<loadSVMvector.size()<<endl;
	//HOGDescriptor hogd;
	hogTest.load("hogSVMFaces.xml");


	vector<Rect> found, found_filtered;
	vector<double> found_weights;

	vector<int> test;
	
	//for (int i = 0; i < test_lst.size(); ++i)
	//{
		stringstream filePathName;
		filePathName << pathData << "/" << "7.jpg";
		
		Mat testImg = imread(filePathName.str(), 1);
		cout<<"img read"<<endl;
		
		vector<Point> found_locations;
		
		int i = 1;
		float a;
		Mat druga;
		descriptors.clear();
		do
		{
			hogTest.winSize = Size(64*i, 64*i);
			hogTest.blockSize = Size(4*i, 4*i);
			hogTest.blockStride = Size(2*i, 2*i);
			hogTest.cellSize = Size(2*i, 2*i);
			cout<<"set parametars"<<endl;
		
			for(int j=0; j+64*i<testImg.rows; j+=32*i)
			{
				cout<<"first for: "<<j<<endl;
				for(int k=0; k+64*i<testImg.cols; k+=32*i)
				{
					cout<<"second for: "<<k<<endl;
					druga = testImg(Range(j, j+64*i), Range(k, k+64*i));
					cout<<druga.rows<<endl;
					cout<<druga.cols<<endl;
					hogTest.compute(druga, descriptors, Size(0, 0), Size(0, 0));
					cout<<"deskriptor size: "<<descriptors.size()<<endl;
					a = svm->predict(descriptors);
					cout<<"detect"<<endl;
					if(a == -1)
						rectangle(testImg, Point(k, j), Point(k+64*i, j+64*i), Scalar(0, 255, 0));
					cout<<"draw rectangle"<<endl;

					descriptors.clear();
					
					
				}
			}
			cout<<"i = "<<i<<endl;
			i++;
		}while(64*i < min(testImg.cols, testImg.rows));
		imshow( "hogSVMFaces.xml", testImg );
	
		
	
	waitKey(0);
}

// Following subroutine "get_svm_detector" to convert SVM parameters to vector floats value has been taken from the mentioned website
// https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;

	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);

	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||

		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));

	CV_Assert(sv.type() == CV_32F);

	hog_detector.clear();
	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

