//
//  main.cpp
//  Reverse image search
//
//  Created by Ralf Cheung on 4/12/14.
//  Copyright (c) 2014 Ralf Cheung. All rights reserved.
//

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Image.h"

using namespace cv;

void readme();
IplImage* convertImageRGBtoHSV(const IplImage *imageRGB);


std::vector<string> saveFileName(char *fileName){
    std::string s = fileName;
    std::string delimiter = "_., ";
    
    size_t pos = 0;
    std::string token;
    std::vector<string> sourceVector;
    
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        sourceVector.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    return sourceVector;
}

/** @function main */
int main( int argc, char** argv )
{
    //300 350
    
    double probability = 0.6;

    std::vector<string> sourceVector = saveFileName(argv[1]);
    std::vector<string> searchVector = saveFileName(argv[2]);

    
    double matched = 0;
    
    for(std::vector<string>::iterator sourceIterator = sourceVector.begin(); sourceIterator != sourceVector.end(); sourceIterator++ ){
        for(std::vector<string>::iterator searchIterator = searchVector.begin(); searchIterator != searchVector.end(); searchIterator++ ){
            if (*sourceIterator == *searchIterator) {
                matched++;
                break;
            }
            
            
        }
    }

    probability *= (matched / sourceVector.size());
    
    
    
    MyImage *image = new MyImage();
    image->setWidth(352);
    image->setHeight(288);
    image->setImagePath(argv[1]);

    if(!image->ReadImage()){
        std::cout << "Error reading MyImage" << std::endl;
    }

    printf("%s", argv[1]);
    
    
//    cv::Mat object = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR );
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        printf("error\n");
        exit(1);
    }
    unsigned char pixels[352 * 288 * 3];
    fread(pixels, sizeof(unsigned char), 352*288 * 3, f);
    fclose(f);
    cv::Mat object(Size(352, 288), CV_8UC3, pixels);
    
    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", object);
    
    if( !object.data )
    {
        std::cout<< "Error reading object " << std::endl;
        return -1;
    }
    
    cv::Mat tmp, alpha;
    threshold(object, object, 255, 0, THRESH_TOZERO_INV);

    cv::Rect myROI(100, 100, 150, 150);
    
    
    //Detect the keypoints using SURF Detector
    int minHessian = 1000;
    
    cv::SurfFeatureDetector detector( minHessian );
    std::vector<cv::KeyPoint> kp_object;
    detector.upright = false;
    detector.detect( object, kp_object );
    
    //Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;
    cv::Mat des_object;
    
    extractor.compute( object, kp_object, des_object );
    
    cv::FlannBasedMatcher matcher;
    
    
    cv::namedWindow("Good Matches");
    
    std::vector<cv::Point2f> obj_corners(4);
    
    //Get the corners from the object
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( object.cols, 0 );
    obj_corners[2] = cvPoint( object.cols, object.rows );
    obj_corners[3] = cvPoint( 0, object.rows );
    
    char key = 'a';
    int framecount = 0;
    while (key != 27)
    {
        
        
        
        
        if (framecount < 5)
        {
            framecount++;
            continue;
        }
        
        cv::Mat des_image, img_matches;
        std::vector<cv::KeyPoint> kp_image;
        std::vector<std::vector<cv::DMatch > > matches;
        std::vector<cv::DMatch > good_matches;
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        std::vector<cv::Point2f> scene_corners(4);
        cv::Mat H;
//        cv::Mat image = cv::imread( argv[2], CV_LOAD_IMAGE_COLOR );
        MyImage *sceneMyPic = new MyImage();
        sceneMyPic->setWidth(352);
        sceneMyPic->setHeight(288);
        sceneMyPic->setImagePath(argv[2]);
        sceneMyPic->ReadImage();

        cv::Mat image(Size(352, 288), CV_8UC3, sceneMyPic->getImageData());

        
        detector.detect( image, kp_image );
        extractor.compute( image, kp_image, des_image );
        
        matcher.knnMatch(des_object, des_image, matches, 2);
        
        for(int i = 0; i < cv::min(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
        {
            if((matches[i][0].distance < 0.8*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
            {
                good_matches.push_back(matches[i][0]);
            }
        }
        
        //Draw only "good" matches
        drawMatches( object, kp_object, image, kp_image, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        if (good_matches.size() >= 4)
        {
            
            Point2f averagePoint;
            for( int i = 0; i < good_matches.size(); i++ )
            {
                //Get the keypoints from the good matches
                obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
                averagePoint.x += kp_image[ good_matches[i].trainIdx ].pt.x;
                averagePoint.y += kp_image[ good_matches[i].trainIdx ].pt.y;
            }
            
            averagePoint.x /= good_matches.size();
            averagePoint.y /= good_matches.size();
            int inRange = 0;
            int delta = 60;
            for( int i = 0; i < good_matches.size(); i++ ){
                int x =kp_image[ good_matches[i].trainIdx ].pt.x;
                int y =kp_image[ good_matches[i].trainIdx ].pt.y;

                if ((x > (averagePoint.x - delta) && x < (averagePoint.x + delta)) && (y > (averagePoint.y - delta) && (y < (averagePoint.y + delta)))) {
                    inRange++;
                }
                
            } 

            if (probability + (double)inRange / good_matches.size() > 0.8) {
                printf("found\n");
            }else{
                MyImage *objectPic = new MyImage();
                objectPic->setWidth(352);
                objectPic->setHeight(288);
                objectPic->setImagePath(argv[1]);
                objectPic->ReadImage();
                
                cv::Mat objectImage(Size(352, 288), CV_8UC3, objectPic->getImageData());
                cv::Mat smallerQueryImage;
                resize(objectImage, smallerQueryImage, Size(16, 16), 0,0, INTER_CUBIC);

                cvtColor(objectImage, objectImage, COLOR_RGB2HSV);

                for (int x = 0; x < 352; x += 16) {
                    for (int y = 0; y < 288; y += 16) {
                        Rect region(Point(x, y), Size(16, 16));
                        
                        cv::Mat subSampleOfScene = image(region);

                        cvtColor(subSampleOfScene, subSampleOfScene, COLOR_RGB2HSV);
                        int h_bins = 50; int s_bins = 60;
                        int histSize[] = { h_bins, s_bins };

                        float h_ranges[] = { 0, 180 };
                        float s_ranges[] = { 0, 256 };
                        
                        const float* ranges[] = { h_ranges, s_ranges };
                        
                        // Use the o-th and 1-st channels
                        int channels[] = { 0, 1 };
                        MatND hist_base;
                        MatND hist_test;
                        calcHist( &smallerQueryImage, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
                        normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

                        calcHist( &subSampleOfScene, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false );
                        normalize( hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat() );
                        double base_test1 = compareHist(hist_base, hist_test, CV_COMP_CORREL);
                        if (base_test1 > 0.1) {
                            probability += base_test1;
                        }

                    }
                }
                
                if (probability > 0.8) { 
                    
                    printf("found with confidence: %f\n", probability);
                }
                else{
                    printf("not found\n");
                }
               
            }
            
            
//            H = findHomography( obj, scene, CV_RANSAC );
//            
//            perspectiveTransform( obj_corners, scene_corners, H);

            
            line(img_matches,  cv::Point2f(averagePoint.x - 40 + sceneMyPic->getWidth(), averagePoint.y - 40), cv::Point2f(averagePoint.x + 40 + sceneMyPic->getWidth(), averagePoint.y - 40), cv::Scalar(0, 255, 0), 4);
            line(img_matches, cv::Point2f(averagePoint.x + 40 + sceneMyPic->getWidth(), averagePoint.y - 40), cv::Point2f(averagePoint.x + 40 + sceneMyPic->getWidth(), averagePoint.y + 40), cv::Scalar(0, 255, 0), 4);
            line(img_matches, cv::Point2f(averagePoint.x + 40 + sceneMyPic->getWidth(), averagePoint.y + 40), cv::Point2f(averagePoint.x - 40 + sceneMyPic->getWidth(), averagePoint.y + 40), cv::Scalar(0, 255, 0), 4);
            line(img_matches, cv::Point2f(averagePoint.x - 40 + sceneMyPic->getWidth(), averagePoint.y + 40), cv::Point2f(averagePoint.x - 40 + sceneMyPic->getWidth(), averagePoint.y - 40), cv::Scalar(0, 255, 0), 4);

            
        }
        imshow("Good matches", img_matches);
        
        key = cv::waitKey(0);
    }
    return 0;

}


IplImage* convertImageRGBtoHSV(const IplImage *imageRGB)
{
	float fR, fG, fB;
	float fH, fS, fV;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
    
	// Create a blank HSV image
	IplImage *imageHSV = cvCreateImage(cvGetSize(imageRGB), 8, 3);
	if (!imageHSV || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
		printf("ERROR in convertImageRGBtoHSV()! Bad input image.\n");
		exit(1);
	}
    
	int h = imageRGB->height;		// Pixel height.
	int w = imageRGB->width;		// Pixel width.
	int rowSizeRGB = imageRGB->widthStep;	// Size of row in bytes, including extra padding.
	char *imRGB = imageRGB->imageData;	// Pointer to the start of the image pixels.
	int rowSizeHSV = imageHSV->widthStep;	// Size of row in bytes, including extra padding.
	char *imHSV = imageHSV->imageData;	// Pointer to the start of the image pixels.
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x*3);
			int bB = *(uchar*)(pRGB+0);	// Blue component
			int bG = *(uchar*)(pRGB+1);	// Green component
			int bR = *(uchar*)(pRGB+2);	// Red component
            
			// Convert from 8-bit integers to floats.
			fR = bR * BYTE_TO_FLOAT;
			fG = bG * BYTE_TO_FLOAT;
			fB = bB * BYTE_TO_FLOAT;
            
			// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
			float fDelta;
			float fMin, fMax;
			int iMax;
			// Get the min and max, but use integer comparisons for slight speedup.
			if (bB < bG) {
				if (bB < bR) {
					fMin = fB;
					if (bR > bG) {
						iMax = bR;
						fMax = fR;
					}
					else {
						iMax = bG;
						fMax = fG;
					}
				}
				else {
					fMin = fR;
					fMax = fG;
					iMax = bG;
				}
			}
			else {
				if (bG < bR) {
					fMin = fG;
					if (bB > bR) {
						fMax = fB;
						iMax = bB;
					}
					else {
						fMax = fR;
						iMax = bR;
					}
				}
				else {
					fMin = fR;
					fMax = fB;
					iMax = bB;
				}
			}
			fDelta = fMax - fMin;
			fV = fMax;				// Value (Brightness).
			if (iMax != 0) {			// Make sure it's not pure black.
				fS = fDelta / fMax;		// Saturation.
				float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
				if (iMax == bR) {		// between yellow and magenta.
					fH = (fG - fB) * ANGLE_TO_UNIT;
				}
				else if (iMax == bG) {		// between cyan and yellow.
					fH = (2.0f/6.0f) + ( fB - fR ) * ANGLE_TO_UNIT;
				}
				else {				// between magenta and cyan.
					fH = (4.0f/6.0f) + ( fR - fG ) * ANGLE_TO_UNIT;
				}
				// Wrap outlier Hues around the circle.
				if (fH < 0.0f)
					fH += 1.0f;
				if (fH >= 1.0f)
					fH -= 1.0f;
			}
			else {
				// color is pure Black.
				fS = 0;
				fH = 0;	// undefined hue
			}
            
			// Convert from floats to 8-bit integers.
			int bH = (int)(0.5f + fH * 255.0f);
			int bS = (int)(0.5f + fS * 255.0f);
			int bV = (int)(0.5f + fV * 255.0f);
            
			// Clip the values to make sure it fits within the 8bits.
			if (bH > 255)
				bH = 255;
			if (bH < 0)
				bH = 0;
			if (bS > 255)
				bS = 255;
			if (bS < 0)
				bS = 0;
			if (bV > 255)
				bV = 255;
			if (bV < 0)
				bV = 0;
            
			// Set the HSV pixel components.
			uchar *pHSV = (uchar*)(imHSV + y*rowSizeHSV + x*3);
			*(pHSV+0) = bH;		// H component
			*(pHSV+1) = bS;		// S component
			*(pHSV+2) = bV;		// V component
		}
	}
	return imageHSV;
}

