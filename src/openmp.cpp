#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Cartesian.h>
#include <omp.h>

using namespace std;


// CGAL Kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay_triangulation;
typedef K::Point_3 Point;

// Function to read color data from a text file
Eigen::MatrixXf readColorData(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<Eigen::Vector3f> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        Eigen::Vector3f point;
        ss >> point[0] >> point[1] >> point[2];
        data.push_back(point);
    }

    file.close();

    Eigen::MatrixXf dataMatrix(data.size(), 3);
    for (size_t i = 0; i < data.size(); ++i) {
        dataMatrix.row(i) = data[i];
    }

    return dataMatrix;
}

// Function to compute barycentric coordinates of a point within a tetrahedron
Eigen::Vector4f compute_barycentric_coordinates(const Point& p, const Point& p0, const Point& p1, const Point& p2, const Point& p3) {
    Eigen::Matrix4f T;
    T << p0.x(), p1.x(), p2.x(), p3.x(),
         p0.y(), p1.y(), p2.y(), p3.y(),
         p0.z(), p1.z(), p2.z(), p3.z(),
         1.0, 1.0, 1.0, 1.0;

    Eigen::Vector4f rhs(p.x(), p.y(), p.z(), 1.0);

    Eigen::Vector4f bary_coords = T.colPivHouseholderQr().solve(rhs);

    return bary_coords;
}

// Function to interpolate color for a given FL range
Eigen::Vector3f interpolate_color(const Eigen::MatrixXf& data, const tuple<int, int, int>& FL_range, const Delaunay_triangulation& dt) {
    Point query_point(get<0>(FL_range), get<1>(FL_range), get<2>(FL_range));

    auto cell = dt.locate(query_point);
    if (dt.is_infinite(cell)) {
        // Handle the case where the point is not within the convex hull
        return Eigen::Vector3f::Zero();
    }

    // Get the vertices of the simplex
    Point p0 = cell->vertex(0)->point();
    Point p1 = cell->vertex(1)->point();
    Point p2 = cell->vertex(2)->point();
    Point p3 = cell->vertex(3)->point();

    // Compute the barycentric coordinates of the query point within the simplex
    Eigen::Vector4f bary_coords = compute_barycentric_coordinates(query_point, p0, p1, p2, p3);

    // Perform interpolation using the barycentric coordinates
    Eigen::Vector3f interpolated_color = Eigen::Vector3f::Zero();

    for (int i = 0; i < 4; ++i) {
        Point p = cell->vertex(i)->point();
        for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
            if (vit->point() == p) {
                interpolated_color += bary_coords[i] * data.row(distance(dt.finite_vertices_begin(), vit)).transpose();
                break;
            }
        }
    }

    return interpolated_color;
}

int main() {
    vector<string> color_names = {"red", "green", "blue", "cyan", "magenta", "yellow", "white", "black"};
    map<string, Eigen::MatrixXf> color_data;

    // Read data for each color
    #pragma omp parallel for
    for (const auto& color : color_names) {
        string filePath = "../input/data/" + color + ".txt"; // Update with actual path
        #pragma omp critical
        color_data[color] = readColorData(filePath);
    }

    // Front lights data
    vector<Point> front_lights = {
        Point(0, 0, 0), Point(0, 0, 128), Point(0, 0, 255),
        Point(0, 128, 0), Point(0, 128, 128), Point(0, 128, 255),
        Point(0, 255, 0), Point(0, 255, 128), Point(0, 255, 255),
        Point(128, 0, 0), Point(128, 0, 128), Point(128, 0, 255),
        Point(128, 128, 0), Point(128, 128, 128), Point(128, 128, 255),
        Point(128, 255, 0), Point(128, 255, 128), Point(128, 255, 255),
        Point(255, 0, 0), Point(255, 0, 128), Point(255, 0, 255),
        Point(255, 128, 0), Point(255, 128, 128), Point(255, 128, 255),
        Point(255, 255, 0), Point(255, 255, 128), Point(255, 255, 255)
    };

    // Prepare Delaunay triangulation
    Delaunay_triangulation dt;
    dt.insert(front_lights.begin(), front_lights.end());

    // Example target points
    vector<tuple<int, int, int>> target_points;
    for (int r = 0; r <= 255; r += 1) {
        for (int g = 0; g <= 255; g += 1) {
            for (int b = 0; b <= 255; b += 1) {
                target_points.push_back(make_tuple(r, g, b));
            }
        }
    }
    // Timer start
    auto start = chrono::high_resolution_clock::now();

    // Interpolate color for target points
    #pragma omp parallel for
    for (int i = 0; i < target_points.size(); ++i) {
        const auto& target_point = target_points[i];
        #pragma omp parallel for
        for (const auto& color : color_names) {
            Eigen::Vector3f interpolated_color = interpolate_color(color_data[color], target_point, dt);
            // cout << "Interpolated color for " << color << " at (" << get<0>(target_point) << ", " << get<1>(target_point) << ", " << get<2>(target_point) << "): " << interpolated_color.transpose() << endl;
        }
    }

    // Timer end
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Runnging openmp..." << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return 0;
}