#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <omp.h>

int main()
{

	constexpr int imNum=2;
	std::vector<std::string> filename { "base.jpg", "locate.jpg"};
        std::vector<cv::Mat> imVec(imNum);
        std::vector<std::vector<cv::KeyPoint>>keypointVec(imNum);
        std::vector<cv::Mat> descriptorsVec(imNum);

        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();  
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
        auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
       for (int i=0; i<imNum; i++){
	  imVec[i] = cv::imread( filename[i], cv::IMREAD_ANYCOLOR);
          detector->detect( imVec[i], keypointVec[i] );
          extractor->compute( imVec[i],keypointVec[i],descriptorsVec[i]);
          std::cout << "find " << keypointVec[i].size() << " keypoints in im" << i << "\n";
       }

        auto end = std::chrono::high_resolution_clock::now();

        cv::BFMatcher brue_force_matcher = cv::BFMatcher(cv::NORM_HAMMING, true);

        std::vector< cv::DMatch > matches;
        brue_force_matcher.match((const cv::OutputArray)descriptorsVec[0], (const cv::OutputArray)descriptorsVec[1],  matches);

	std::sort(matches.begin(), matches.end(), [](auto a, auto b) {return a.distance < b.distance; });

        if (matches.size() > 10)
        {
                matches.resize(10);
        }

        cv::Mat output_image;

        for (int i = 0; i < matches.size(); i++)
        {
                std::cout << matches[i].distance << ", ";
        }
        std::cout << std::endl;

 	std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";

        cv::drawMatches(                imVec[0], keypointVec[0],
                                        imVec[1], keypointVec[1],
                                        matches,
                                        output_image);

//      cv::imshow("Matches", output_image);
//      cv::waitKey(0);
	cv::imwrite("matches.png", output_image);
        return 0;
}
