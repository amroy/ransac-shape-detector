/*
 Copyright (c) 2020, Hassan Rami (hassan.rami@outlook.com)
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "ransac.h"
#include <random>
#include <algorithm>
#include <ctime>

Eigen::MatrixX3f slice_rows(const Eigen::MatrixX3f& matrix, const std::vector<int>& rows) {
	Eigen::MatrixX3f sliced_mat;
	sliced_mat.resize(rows.size(), Eigen::NoChange);
	for(int i=0; i<rows.size(); ++i) {
		sliced_mat(i, 0) = matrix(rows[i], 0);
		sliced_mat(i, 1) = matrix(rows[i], 1);
		sliced_mat(i, 2) = matrix(rows[i], 2);
	}

	return sliced_mat;
}

/**
 * PlaneModel
 */
PlaneModel::PlaneModel(float eps, float alp): epsilon(eps), alpha(alp) { 
    n = 1; 
}

bool PlaneModel::fit(const Eigen::MatrixX3f& points, const Eigen::MatrixX3f& normals) {
    if(points.rows() == 3) {
        const float *row_data = points.row(0).data();
        Eigen::Vector3f p1 = points.row(0);
        Eigen::Vector3f p2 = points.row(1);
        Eigen::Vector3f p3 = points.row(2);
        Eigen::Vector3f v1 = p1 - p2;
        Eigen::Vector3f v2 = p1 - p3;
        this->normal_vector = v1.cross(v2);
        this->base_points = points;
    } else {
        this->normal_vector = normals.row(0);
        this->base_points = points;
    }

    return true;
}

bool PlaneModel::evaluate(const Eigen::MatrixX3f& points, std::vector<int> &inliers, std::vector<int> &outliers) {
    float distance{0.0};
    // Profject every point on plane normal vector to get gett distances to plane
    for (int i = 0; i < points.rows(); ++i)
    {
        distance = (points.row(i) - base_points.row(0)).dot(normal_vector.transpose());
        if(distance <= epsilon && distance >= -epsilon)
            inliers.push_back(i);
        else {
            outliers.push_back(i);
        }
    }

    return true; 
}

/**
 * SphereModel
 */
SphereModel::SphereModel(float eps, float alp, float min_rad, float max_rad)
    : epsilon(eps), alpha(alp), min_radius(min_rad), max_radius(max_rad) {
    n = 2;
}

bool SphereModel::fit(const Eigen::MatrixX3f& points, const Eigen::MatrixX3f& normals) {

    return false;
}

bool SphereModel::evaluate(const Eigen::MatrixX3f& points, std::vector<int> &inliers, std::vector<int> &outliers) {
    
    return false;
}

/**
 * RANSAC algorithm
 */
RANSAC::RANSAC(Eigen::MatrixX3f &_points, Eigen::MatrixX3f &_normals)
    : points(_points), normals(_normals) {}

RANSAC::RANSAC(Eigen::MatrixX3f &_points, Eigen::MatrixX3f &_normals, const int n, const int m)
    : points(_points), normals(_normals), niter(n), min_samples(m) {}

void RANSAC::sample() {
    size_t N = points.rows();
    samples_ind.clear();
    std::srand((unsigned) time(0));
    for(size_t i=0; i<model->n; ++i)
        samples_ind.push_back((std::rand() % N));
}

bool RANSAC::run() {
    int nb_inliers{0}, k{0};
    bool success{false};
    while(k <= niter) {
        sample();
        Eigen::MatrixX3f sampled_points = std::move(slice_rows(points, samples_ind));
        Eigen::MatrixX3f sampled_normals = std::move(slice_rows(normals, samples_ind));
        if(model->fit(sampled_points, sampled_normals)) {
            std::vector<int> _inliers, _outliers;
            if(model->evaluate(points, _inliers, _outliers)) {
                if(_inliers.size() > min_samples && _inliers.size() > nb_inliers) {
                    nb_inliers = _inliers.size();
                    inliers = std::move(_inliers);
                    outliers = std::move(_outliers);
                    success = true;
                }
            }
        }
        k++;
    }
    if(success) {
        std::cout << "Model " << model->name << " fitted with " << nb_inliers << " inliers." << std::endl;
    } else {
        std::cout << "Could not find any points that fit the " << model->name << " model." << std::endl;
    }

    return success;
}

void RANSAC::setModel(std::unique_ptr<IModel>& _model) {
    this->model = std::move(_model);
}

IModel& RANSAC::getModel() {
    return *model;
}

std::vector<int> RANSAC::getInliers() const {
    return inliers;
}

Eigen::MatrixX3f RANSAC::getInliersCoords() const {
    Eigen::MatrixX3f inliers_coords = slice_rows(points, inliers);
    return inliers_coords;
}

std::vector<int> RANSAC::getOutliers() const {
    return outliers;
}

Eigen::MatrixX3f RANSAC::getOutliersCoords() const {
    Eigen::MatrixX3f _coords = slice_rows(points, outliers);
    return _coords;
}
RANSAC::~RANSAC() {
    std::cout << "RANSAC destroyed" << std::endl;
}
