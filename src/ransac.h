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

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>

/**
 * This function takes only a subset of the rows and returns a sliced matrix
 */
Eigen::MatrixX3f slice_rows(const Eigen::MatrixX3f& matrix, const std::vector<int>& rows);

struct IModel {
	virtual bool fit(const Eigen::MatrixX3f&, const Eigen::MatrixX3f&) = 0;
	virtual bool evaluate(const Eigen::MatrixX3f&, std::vector<int> &inliers, std::vector<int> &outliers) = 0;
	int n{ 1 };
	std::string name;
};

class PlaneModel : public IModel {
public:
	PlaneModel() = default;
	PlaneModel(float eps, float alp);
	bool fit(const Eigen::MatrixX3f&, const Eigen::MatrixX3f&) override ;
	bool evaluate(const Eigen::MatrixX3f& points, std::vector<int> &inliers, std::vector<int> &outliers) override ;
private:
	const float epsilon{ 2.0 };
	const float alpha{ 0.15 };
	
	Eigen::Vector3f normal_vector{ .0, .0, .0 };
	Eigen::MatrixXf base_points;
};

class SphereModel : public IModel {
public:
	SphereModel() = default;
	SphereModel(float epsilon, float alpha, float min_radius = 20.0, float max_radius = 100.0);
    bool fit(const Eigen::MatrixX3f&, const Eigen::MatrixX3f&) override ;
    bool evaluate(const Eigen::MatrixX3f& points, std::vector<int> &inliers, std::vector<int> &outliers) override ;
private:
	const float epsilon{ 2.0 };
	const float alpha{ 0.15 };
	const float min_radius{ 20.0 };
	const float max_radius{ 100.0 };
};

class RANSAC {
public:
	RANSAC() = delete;
	RANSAC(Eigen::MatrixX3f&, Eigen::MatrixX3f&);
	RANSAC(Eigen::MatrixX3f&, Eigen::MatrixX3f&, const int, const int);
	~RANSAC();

	void setModel(std::unique_ptr<IModel>&);
	IModel& getModel();

	std::vector<int> getInliers() const;
	Eigen::MatrixX3f getInliersCoords() const;
	
	std::vector<int> getOutliers() const;
	Eigen::MatrixX3f getOutliersCoords() const;

	void sample();
    bool run();

private:
	const int niter{ 100 };
	const int min_samples{ 30 };
	std::vector<int> samples_ind;
	Eigen::MatrixX3f points;
	Eigen::MatrixX3f normals;
	std::unique_ptr<IModel> model{nullptr};
	std::vector<int> inliers;
	std::vector<int> outliers;
};
