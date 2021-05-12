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


#define HOW_LONG 0 // wait until key or X is pressed, other value is in milisecs

using namespace std;
using namespace cv;
using namespace cv::ml;

vector< Mat > img_pos_lst, img_neg_list;
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData );
Mat get_hogdescriptor_visual_image(Mat& origImg, vector<float>& descriptorValues, Size winSize, Size cellSize, int scaleFactor, double viz_factor);

int main(int argc, const char* argv[])
{
	string pathData;
	int posCount, negCount;
	//cout << "Enter Positive Data directory path" << endl;
	pathData = "/home/nemanja/Desktop/FaceDetection";

	//Positive image count
	posCount = 5417;
	//Negative image count
	negCount = 5417;
	
	int in = 0;
	

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
		//imshow("read from pveimages",img);
		//waitKey(HOW_LONG);

		resize(img, img, Size(256, 256));
		//imshow("testPositive", img);
		//waitKey(0);
		
		//imshow("pveimages resized",img);
		//waitKey(HOW_LONG);
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
		
		//imshow("read from nveimages",img);
		//waitKey(HOW_LONG);

		resize(img, img, Size(256, 256));
		//imshow("testNegative", img);
		//waitKey(0);
		
		//imshow("nveimages resized",img);
		//waitKey(HOW_LONG);

		img_neg_list.push_back(img.clone());
	}

	cout<<"Get out of TrainNeg"<<endl;
	
	//Mat img;
	//img = imread("Lenna.png");
	//cvtColor(img, img, CV_RGB2GRAY);
	/*resize(img,img,Size(50,50));*/

	HOGDescriptor hog;
	//cout<<"HOGDescriptor"<<endl;
	Mat gradMat;
	vector<int> labelsMat;
	Mat trainingDataMat;
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
	cout<<"before making pos feature"<<endl;
	for (int i = 0; i < img_pos_lst.size(); ++i)
	{
		hog.compute(img_pos_lst[i], descriptors, Size(0, 0), Size(0, 0));
		//cout<<descriptors.size()<<endl;	
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);		
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		
		
		//Mat background = Mat::zeros(Size(256, 256),CV_8UC1);
		//Mat image = get_hogdescriptor_visual_image(background,descriptors,hog.winSize,hog.cellSize,3, 2.5);
		//imshow("hog of pveimages",image);
		//waitKey(HOW_LONG);
		
		
		
		//int labels=  {1} ;
		//cout<<labels<<endl;
		//Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(-1);
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
	
	//Mat background = Mat::zeros(Size(256, 256),CV_8UC1);
	
	//for negative data
	for (int i = 0; i < img_neg_list.size(); ++i)
	{
		hog.compute(img_neg_list[i], descriptors, Size(0, 0), Size(0, 0));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		
		
		
		//Mat image = get_hogdescriptor_visual_image(background,descriptors,hog.winSize,hog.cellSize,3, 2.5);
		//imshow("hog of nveimages",image);
		//waitKey(HOW_LONG);
		
		
		
		//int labels = {-1};
		//cout<<labels<<endl;
		//Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(1);
	}
	
	
	cout<<"neg_lst"<<endl;

	// Set up training data
	//int labels[2] = { 1, 0 };
	//Mat labelsMat(2, 1, CV_32S, labels);

	// Set up SVM's parameters
	Ptr<SVM> svm = SVM::create();
	
	//svm->setCoef0( 0.0 );
    	 
        
        //svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
        svm->setType(SVM::C_SVC);
        svm->setKernel( SVM::LINEAR );
        svm->setGamma( 0 ); //3
        svm->setDegree( 1 );//3
    	svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    	svm->setNu( 0.5 );
    	svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    	svm->setC( 0.01 ); // From paper, soft classifier
	svm->setType(SVM::C_SVC);
	cout<<"set svm coef"<<endl;
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//Mat train_data;
	//cout<<"created train_data"<<endl;
	//convert_to_ml(trainingDataMat, train_data);
	//cout<<"converted to ml"<<endl;
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	// Train the SVM with given parameters
	cout<<trainingDataMat.size()<<endl;
	//transpose(labelsMat, labelsMat);
	//cout<<labelsMat.size()<<endl;
	//Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	//		cout << "tako nesto 1"<<endl;
	//svm->trainAuto(td);
	//				cout << "tako nesto 2"<<endl;
	//cout<<"pre"<<endl;
	
	
	//transpose(trainingDataMat, trainingDataMat);
	//transpose(labelsMat, labelsMat);
	/*
	cout<<"cgrig"<<endl;
	Ptr< ParamGrid >  	Cgrid = SVM::getDefaultGridPtr(SVM::C);
	cout<<"gammagrid"<<endl;
	Ptr< ParamGrid >  	gammaGrid = SVM::getDefaultGridPtr(SVM::GAMMA);
	cout<<"pgrid"<<endl;
	Ptr< ParamGrid >  	pGrid = SVM::getDefaultGridPtr(SVM::P);
	cout<<"nugrid"<<endl;
	Ptr< ParamGrid >  	nuGrid = SVM::getDefaultGridPtr(SVM::NU);
	cout<<"coeffgrid"<<endl;
	Ptr< ParamGrid >  	coeffGrid = SVM::getDefaultGridPtr(SVM::COEF);
	cout<<"degreegrid"<<endl;
	Ptr< ParamGrid >  	degreeGrid = SVM::getDefaultGridPtr(SVM::DEGREE);	
	cout<<"befor train"<<endl;  	
	svm->trainAuto(trainingDataMat, ROW_SAMPLE, labelsMat, 10, SVM::getDefaultGridPtr(SVM::C), SVM::getDefaultGridPtr(SVM::GAMMA), SVM::getDefaultGridPtr(SVM::P), SVM::getDefaultGridPtr(SVM::NU), SVM::getDefaultGridPtr(SVM::COEF), SVM::getDefaultGridPtr(SVM::DEGREE), 1);
	//svm->train(td);*/
	cout<<"trained"<<endl;
	//svm->save("hogSVMFaces.xml");


	//hog.compute(img, descriptors, Size(10, 10));
	//float prediction = svm->predict(descriptors);
	//cout<<"svm"<<endl;
	Ptr<SVM> svmLoad;
	//svmLoad = SVM::load("hogSVMFaces.xml");
	//Mat loadSVMMat = svmLoad->getSupportVectors();
	vector<float> loadSVMvector;
	
	cout<<"load svm"<<endl;

	get_svm_detector(svm, loadSVMvector);
	HOGDescriptor hogTest;
	hogTest.winSize = Size(256, 256);
	hogTest.blockSize = Size(16, 16);
	//hogTest.blockStride = Size(2, 2);
	hogTest.cellSize = Size(8, 8);
	//hogTest.nbins = 9;
	hogTest.setSVMDetector(loadSVMvector);
	hogTest.save("hogSVMFaces.xml");
	cout<<loadSVMvector.size()<<endl;
	//HOGDescriptor hogd;
	hogTest.load("hogSVMFaces.xml");


	vector<Rect> found, found_filtered;
	vector<double> found_weights;
	//Test image name to enter
	//Mat testImg = imread("1.png",0);


	//cout << "Enter Tes' Data directory path" << endl;
	//cin >> pathData;
	int test_pic_count = 17;
	
	for (int i = 1; i < (test_pic_count + 1); ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "/" <<i<< ".png";
		
		Mat testImg = imread(filePathName.str(), 1);
		resize(testImg, testImg, Size(256,256));
		//resize(testImg, testImg, Size(256, 256));
		
		
		imshow("test picture",testImg);
			waitKey(HOW_LONG);
		
		//cout<< "nesto posle"<<endl;
		//resize(testImg, testImg, Size(50, 50));
		//HOG detection function
		
		//hogd.detectMultiScale(testImg, found, found_weights, 0, Size(32,32), Size(0, 0), 1.15, 3, 0);
		
		hogTest.compute(testImg, descriptors, Size(0, 0), Size(0, 0));
		cout<<descriptors.size()<<endl;
		/*Mat image = get_hogdescriptor_visual_image(background,descriptors,hog.winSize,hog.cellSize,3, 2.5);
		imshow("hog of test image",image);
		waitKey(HOW_LONG);*/
		/*
		hogTest.detectMultiScale(testImg, found, found_weights, 0, Size(32,32), Size(0, 0), 1.15, 3, 0);
		
		 for ( size_t j = 0; j < found.size(); j++ )
		{
		    Scalar color = Scalar( 0, found_weights[j] * found_weights[j] * 200, 0 );
		    rectangle( testImg, found[j], color, testImg.cols / 400 + 1 );
		}
		//resize(testImg, testImg, Size(256, 256));
		imshow( "hogSVMFaces.xml", testImg );
	*/
		//cout<<found.size()<<endl;
		
		vector<Point> found_locations;
		hogTest.detect(testImg, found_locations, 0);
		cout<<found_locations.size()<<endl;
		
		cout <<"Test image no "<<i<<endl;
		
		if(!found_locations.empty())
		{
	    		cout<<"PERSON"<<endl; 
		}
		else
			cout<<"NOT PERSON"<<endl;
			 
	}
			
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
    cout<<"cols"<<train_samples[0].cols;
    cout<<"rows"<<train_samples[0].rows;
    cout<<"cols"<<cols<<endl;
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

Mat get_hogdescriptor_visual_image(Mat& origImg,
	vector<float>& descriptorValues, //hog feature vector
	 Size winSize, // ​​picture window size
	Size cellSize,
	 int scaleFactor, // ​​scale the proportion of the background image
	 double viz_factor)//scaling the line length ratio of the hog feature
{
	 Mat visual_image; // finally visualized image size
	resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
 
	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	 float radRangeForOneBin = 3.14 / (float)gradientBinSize; //pi=3.14 corresponds to 180°
 
	// prepare data structure: 9 orientation / gradient strenghts for each cell
	 int cells_in_x_dir = origImg.rows / cellSize.width; // number of cells in the x direction
	 int cells_in_y_dir = origImg.cols / cellSize.height;//number of cells in the y direction
	 int totalnrofcells = cells_in_x_dir * cells_in_y_dir; // total number of cells
	 // Note the definition of the three-dimensional array here
	//int ***b;
	//int a[2][3][4];
	//int (*b)[3][4] = a;
	//gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;
 
			for (int bin = 0; bin<gradientBinSize; bin++)
				 gradientStrengths[y][x][bin] = 0.0;//Initialize the gradient strength corresponding to the 9 bins of each cell to 0
		}
	}
 
	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	 //equivalent to blockstride = (8,8)
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;
 
	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;
 
	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}
 
				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;
 
					 gradientStrengths[celly][cellx][bin] += gradientStrength;//because C is stored in rows
 
				} // for (all bins)
 
 
				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				 cellUpdateCounter[celly][cellx]++;//Because there is overlap between blocks, it is necessary to record which cells are calculated multiple times.
 
			} // for (all cells)
 
 
		} // for (all block x pos)
	} // for (all block y pos)
 
 
	// compute average gradient strengths
	for (int celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
 
			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}
 
 /*
	cout << "winSize = " << winSize << endl;
	cout << "cellSize = " << cellSize << endl;
	cout << "blockSize = " << cellSize * 2 << endl;
	cout << "blockNum = " << blocks_in_x_dir << "×" << blocks_in_y_dir << endl;
	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 */
	// draw cells
	for (int celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;
 
			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height /2;
 
			rectangle(visual_image,
				Point(drawX*scaleFactor, drawY*scaleFactor),
				Point((drawX + cellSize.width)*scaleFactor,
				(drawY + cellSize.height)*scaleFactor),
				 CV_RGB (0, 0, 0), // ​​cell frame color
				1);
 
			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
				// no line to draw?
				if (currentGradStrength == 0)
					continue;
 
				 float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2; // take the intermediate value in each bin, such as 10 °, 30 °, ..., 170 °.
 
				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better
 
				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
				// draw gradient visual_imagealization
				line(visual_image,
					Point(x1*scaleFactor, y1*scaleFactor),
					Point(x2*scaleFactor, y2*scaleFactor),
					 CV_RGB (255, 255, 255), // ​​HOG visualized cell color
					1);
 
			} // for (all bins)
 
		} // for (cellx)
	} // for (celly)
 
 
	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;
 
	 return visual_image;//return the final HOG visualization image
 
}

