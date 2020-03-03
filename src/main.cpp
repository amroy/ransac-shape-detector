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

#include <iostream>
#include "ransac.h"
#include "stl_reader.h"
#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include<vtkPolyDataNormals.h>
#include<vtkPointData.h>

int main()
{	
	std::string object_filename{"data/wheel.stl"};
	try {
		vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
		reader->SetFileName(object_filename.c_str());
		reader->Update();
		auto data = reader->GetOutput()->GetPoints();
		const int Np = static_cast<const int>(data->GetNumberOfPoints());
		Eigen::MatrixX3f points(Np, 3);
        Eigen::MatrixX3f normals(Np, 3);

		for (int i = 0; i < Np; i++)
		{
			points(i, 0) = *(data->GetPoint(i));
			points(i, 1) = *(data->GetPoint(i) + 1);
			points(i, 2) = *(data->GetPoint(i) + 2);
		}

		std::cout << "Vertices loaded : " << points.rows() << std::endl;

		vtkSmartPointer<vtkPolyDataNormals> normals_generator = vtkSmartPointer<vtkPolyDataNormals>::New();
		normals_generator->SetInput(reader->GetOutput());
		normals_generator->ComputeCellNormalsOn();
		normals_generator->ComputePointNormalsOn();
		normals_generator->Update();

		std::cout << "Computing normals..." << std::endl;
		const auto normals_data = normals_generator->GetOutput()->GetPointData()->GetNormals();
		for (int i = 0; i < Np; i++)
		{
			normals(i, 0) = *(normals_data->GetTuple(i));
			normals(i, 1) = *(normals_data->GetTuple(i) + 1);
			normals(i, 2) = *(normals_data->GetTuple(i) + 2);
		}

		std::unique_ptr<RANSAC> ransac = std::make_unique<RANSAC>(points, normals, 100, 30);
		std::unique_ptr<IModel> sphere_model = std::make_unique<SphereModel>(2.0, 0.15, 30.0, 100.0);
		std::unique_ptr<IModel> plane_model = std::make_unique<PlaneModel>(2.0, 0.15);
		ransac->setModel(plane_model);
		bool fitted = ransac->run();
		if(fitted) {
			// Visualize inliers
			const Eigen::MatrixX3f inliers = std::move(ransac->getInliersCoords());
			std::cout << "Model fitted successfully with " << inliers.rows() << " points." << std::endl;
		}
	} catch(std::exception& ex) {
		std::cout << ex.what() << std::endl;
	}

	return 0;
}
