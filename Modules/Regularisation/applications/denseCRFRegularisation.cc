/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>

#include <irtkImage.h>
#include <irtkFileToImage.h>

#include "densecrf.h"
#include "common.h"

// Default filenames
char *unary_name = NULL, *image_name = NULL, *outlabel_name = NULL;

void usage(){
	std::cerr << "Usage: denseCRFRegularisation [unary] [image] [outlabel] <options>\n" << std::endl;
	std::cerr << "where <options> is one or more of the following:\n" << std::endl;
	std::cerr << "<-unary>                   Store the displacement field in terms of image coordinate" << std::endl;
    std::cerr << "<-image>                   Store the displacement field in terms of image coordinate" << std::endl;
	std::cerr << "<-outlabel>                Store the total displacement in image" << std::endl;
	exit(1);
}

MatrixXf compute_log_likelihood (irtkRealImage prob, int n_classes){

    double epsilon = 1e-10;

    int sz_img = prob.GetX()*prob.GetY()*prob.GetZ();

    MatrixXf unary (n_classes, sz_img);

    for(int i = 0; i < sz_img; i++){

        // sum up the foreground probabilities to compute the background
        float sum_prob = 0;

        // iterate through all provided foreground prob images
        for(int c = 1; c < n_classes; c++ ){
            // compute ll from prob for all foreground classes:
            float p = prob.GetAsDouble(i) + epsilon;
            unary(c, i) = -log(p);
            sum_prob += p;
        }
        // compute the background probability, bound it to zero and add epsilon for ll
        float p = min(1.0f - sum_prob, 0.0f) + epsilon;
        unary (0, i) = -log(p);
    }

    return unary;
}

void normalise_image_to_uchar(irtkRealImage img, unsigned char* norm_img, int n_channels){

    // create a 1D index to iterate the norm_img array
    double i_min = -1;
    double i_max = -1;

    img.GetMinMaxAsDouble(i_min, i_max);


    for(int i = 0; i < img.GetNumberOfVoxels(); i++) {
        for (int c = 0; c < n_channels; c++) {

            unsigned  int idx_c = c + i * n_channels;

            // normalise to uchar [0,255]
            float curr_intensity = img.GetPointerToVoxels()[i];
            norm_img[idx_c] = ((curr_intensity - i_min) / (i_max - i_min)) * 255.0;

        }
    }
}

int main( int argc, char* argv[]){

    if (argc < 3) {
        usage();
        return 0;
    }

    unary_name  = argv[1];
    argc--;
    argv++;
    image_name  = argv[1];
    argc--;
    argv++;
    outlabel_name  = argv[1];
    argc--;
    argv++;

    // Read unary terms
    std::cout << "Reading unary terms..." << unary_name << std::endl;
    irtkRealImage i_unary(unary_name);

    std::cout << "Reading input image..." << image_name << std::endl;
    irtkRealImage i_image(image_name);

	int n_channels = 1;
	int n_classes = 2;
	int dim[3] = {1,1,1};

    dim[0] = i_unary.GetX();
    dim[1] = i_unary.GetY();
    dim[2] = i_unary.GetZ();

    std::cout << "Initialising fully connected CRF ("
    << dim[0] << "," << dim[1] << "," << dim[2] << ")" << std::endl;
    DenseCRF3D crf(dim[0], dim[1], dim[2], n_classes);

    std::cout << "Computing log-likelihood unary cost..." << std::endl;
    MatrixXf unary = compute_log_likelihood(i_unary, n_classes);
    crf.setUnaryEnergy(unary);

    std::cout << "Normalising image for similarity penality term..." << std::endl;
    // create array with size (n_channels * size_of_image) and normalise to uchar [0,255]
    unsigned char* norm_img = new unsigned char[n_channels*dim[0]*dim[1]*dim[2]];
    normalise_image_to_uchar(i_image, norm_img, n_channels);

    // set the regularisation parameters for the appearance kernel (theta_alpha):
    float theta_nearness[3];
    theta_nearness[0] = 1.0f; theta_nearness[1] = 1.0f; theta_nearness[2] = 1.0f;

    // set the regularisation parameters for the similarity kernel (theta_beta):
    float theta_similarity = 1.0f;

    // set the regularisation parameters for the smoothness kernel (theta_gamma):
    float theta_smoothness[3];
    theta_smoothness[0] = 1.0f; theta_smoothness[1] = 1.0f; theta_smoothness[2] = 1.0f;

    // set the weighting parameters to weight w1 * f(nearness, similarity) + w2 * f(smoothness):
    float w1 = 1.0f;
    float w2 = 1.0f;

    // print the parameters:
    std::cout << "Regularisation term: w1 * f(nearness, similarity) + w2 * f(smoothness)" << std::endl;
    std::cout << "w1 = " << w1 << std::endl;
    std::cout << "Nearness stdev: theta_alpha(x,y,z) = " << theta_nearness[0] << ", " << theta_nearness[1] << ", "
    << theta_nearness[2] << std::endl;
    std::cout << "Similarity stdev: theta_beta(ch_1,...,ch_n) = " << theta_similarity << std::endl;
    std::cout << "Smoothness stdev: theta_gamma(x,y,z) = " << theta_smoothness[0] << ", " << theta_smoothness[1] << ", "
    << theta_smoothness[2] << std::endl;

    // add the Gaussian pairwise and bilateral regularisation terms and weight them by w1, w2
    crf.addPairwiseBilateral1Mod( theta_nearness[0],
                                  theta_nearness[1],
                                  theta_nearness[2],
                                  theta_similarity,
                                  norm_img,
                                  new PottsCompatibility( w1 ) );

    crf.addPairwiseGaussian( theta_smoothness[0], theta_smoothness[1], theta_smoothness[2], new PottsCompatibility( w2 ) );

    // run inference with [n_iter] iterations
    int n_iter = 5;
    MatrixXf out_prob = crf.inference(n_iter);

    // get the discrete label map from the output probabilities
    VectorXs out_seg = crf.currentMap(out_prob);

    // create output label image
    irtkImageAttributes attr = i_unary.GetImageAttributes();
    irtkRealImage i_outlabel(attr);

    // convert to irtk image and write to file
    for(unsigned int i = 0; i < i_outlabel.GetNumberOfVoxels(); i++){

        irtkRealPixel* l = i_outlabel.GetPointerToVoxels();
        l[i] = out_seg(i);
    }

    //i_outlabel.Write(outlabel_name);
    i_outlabel.Write(outlabel_name);


    delete [] norm_img;

}
