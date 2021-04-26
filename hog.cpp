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

using namespace std;
using namespace cv;
using namespace cv::ml;

vector< Mat > img_pos_lst, img_neg_list;
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData );

int main(int argc, const char* argv[])
{
	string pathData;
	int posCount, negCount;
	cout << "Enter Positive Data directory path" << endl;
	cin >> pathData;

	//Positive image count
	posCount = 5417;
	//Negative image count
	negCount = 5417;

	for(int i=1; i < posCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/"<< "pveimages" << "/" << i << ".png";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(),1);
		if (img.empty())
		{ 
			return -1;
		}
		resize(img, img, Size(256, 256));
		//imshow("testPositive", img);
		//waitKey(0);

		img_pos_lst.push_back(img.clone());
	}
	
	cout<<"Get out"<<endl;

	for (int i = 1; i < negCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/" << "nveimages" << "/" << "TrainNeg" <<"/"<<i<< ".png";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(), 1);
		if (img.empty())
		{
			return -1;
		}
		resize(img, img, Size(256, 256));
		//imshow("testNegative", img);
		//waitKey(0);

		img_neg_list.push_back(img.clone());
	}

	cout<<"Get out of TrainNeg"<<endl;
	
	//Mat img;
	//img = imread("Lenna.png");
	//cvtColor(img, img, CV_RGB2GRAY);
	/*resize(img,img,Size(50,50));*/

	HOGDescriptor hog;
	//cout<<"HOGDescriptor"<<endl;
	Mat gradMat,  labelsMat;
	vector<Mat> trainingDataMat;
	std::vector<float> descriptors;

	hog.winSize = Size(256, 256);
	hog.blockSize = Size(16, 16);
	hog.cellSize = Size(8, 8);

	//cout<<"Before HOG.compute"<<endl;
	/*hog.compute(img, descriptors, Size(10, 10));
	Mat descMat = Mat(descriptors);
	transpose(descMat,descMat);
	trainingDataMat.push_back(descMat);*/

	//For positive Data
	cout<<"before msking pos feature"<<endl;
	for (int i = 0; i < img_pos_lst.size(); ++i)
	{
		hog.compute(img_pos_lst[i], descriptors, Size(0, 0), Size(0, 0));
		//cout<<descriptors.size()<<endl;
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		int labels[1] = { 1 };
		Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(temMat);
		//cout<<i<<endl;
	}
	cout<<"after making pos feature"<<endl;
	//cout<<"pos_lst"<<endl;

	//descriptors.clear();
	//descMat.release();
	/*Mat imgNeg = Mat::zeros(Size(50, 50), CV_8UC1);
	hog.compute(imgNeg, descriptors, Size(10, 10));
	descMat = Mat(descriptors);
	transpose(descMat, descMat);
	trainingDataMat.push_back(descMat);*/

	//for negative data
	for (int i = 0; i < img_neg_list.size(); ++i)
	{
		hog.compute(img_neg_list[i], descriptors, Size(0, 0), Size(0, 0));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		int labels[1] = { 0 };
		Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(temMat);
	}
	
	cout<<"neg_lst"<<endl;

	// Set up training data
	//int labels[2] = { 1, 0 };
	//Mat labelsMat(2, 1, CV_32S, labels);

	// Set up SVM's parameters
	Ptr<SVM> svm = SVM::create();
	
	svm->setCoef0( 0.0 );
    	svm->setDegree( 3 );
        svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
        svm->setGamma( 0 );
    	svm->setKernel( SVM::LINEAR );
    	svm->setNu( 0.5 );
    	svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    	svm->setC( 0.01 ); // From paper, soft classifier
	svm->setType(SVM::C_SVC);
	cout<<"set svm coef"<<endl;
	//svm->setKernel(SVM::LINEAR);
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	Mat train_data;
	cout<<"created train_data"<<endl;
	convert_to_ml(trainingDataMat, train_data);
	cout<<"converted to ml"<<endl;
	/*
	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->train(td);
	//cout<<"pre"<<endl;
	*/
	svm->train( train_data, ROW_SAMPLE, labelsMat );
	svm->save("hogSVMFaces.xml");


	//hog.compute(img, descriptors, Size(10, 10));
	//float prediction = svm->predict(descriptors);
	//cout<<"svm"<<endl;
	Ptr<SVM> svmLoad;
	svmLoad = SVM::load("hogSVMFaces.xml");
	//Mat loadSVMMat = svmLoad->getSupportVectors();
	vector<float> loadSVMvector;
	
	cout<<"load svm"<<endl;

	get_svm_detector(svmLoad, loadSVMvector);
	HOGDescriptor hogTest;
	hogTest.winSize = Size(64, 64);
	hogTest.blockSize = Size(4, 4);
	hogTest.blockStride = Size(2, 2);
	hogTest.cellSize = Size(2, 2);
	hogTest.nbins = 9;
	hogTest.setSVMDetector(loadSVMvector);
	hogTest.save("hogSVMFaces.xml");
	hogTest.load("hogSVMFaces.xml");


	vector<Rect> found, found_filtered;
	vector<double> found_weights;
	//Test image name to enter
	//Mat testImg = imread("1.png",0);


	cout << "Enter Tes' Data directory path" << endl;
	cin >> pathData;

	stringstream filePathName;
        filePathName << pathData << "/" << "soprano.png";
        cout << filePathName.str() << endl;
        Mat testImg = imread(filePathName.str(), 1);
        resize(testImg, testImg, Size(1024, 512));
	cout<< "nesto posle"<<endl;
	//resize(testImg, testImg, Size(50, 50));
	//HOG detection function
	
	hogTest.detectMultiScale(testImg, found, found_weights, 0, Size(32,32), Size(0, 0), 1.15, 3, 0);
	
	 for ( size_t j = 0; j < found.size(); j++ )
        {
            Scalar color = Scalar( 0, found_weights[j] * found_weights[j] * 200, 0 );
            rectangle( testImg, found[j], color, testImg.cols / 400 + 1 );
        }
        //resize(testImg, testImg, Size(256, 256));
        imshow( "hogSVMFaces.xml", testImg );

	//cout<<found.size()<<endl;
	/*
	vector<Point> found_locations;
	hogTest.detect(testImg, found_locations);
	cout<<found_locations.size()<<endl;
	if(!found_locations.empty())
	{
    		cout<<"PERSON"<<endl; 
	}
	else
		cout<<"NOT PERSON"<<endl;
	*/	
	//cout<< "multiscale"<<endl;
	/*
	size_t i, j;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	*/

	/*Draw rectangle around detections*/
	/*
	for (i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.7);
		r.y += cvRound(r.height*0.05);
		r.height = cvRound(r.height*0.8);
		rectangle(testImg, r.tl(), r.br(), cv::Scalar(0, 255, 0), 1);
	}

	imshow("testImage",testImg);
	*/
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
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}


