#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <thread>
#include <future>
#include <mutex>
#include <sstream>

using namespace std;

// CGAL Kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
typedef K::Point_2 Point;
typedef std::map<Point, K::FT> Coord_map;

std::string trim_quotes(const std::string& str);
vector<int> interpolate_color_for_color(tuple<int, int, int> FL_range, const vector<Point>& points, const vector<vector<int>>& colors);
vector<vector<double>> readFrontLightCombinations(const string& file_path);
map<string, Eigen::MatrixXf> readColorMeasurementData(const string& filePath);

int main() {
    string dataFilePath = "../input/data/i1_8colors_27FL_v1.csv";
    // Read the color measurement data from the file and process Delaunay triangulation to interpolate colors
    auto FL_pts_all = readFrontLightCombinations(dataFilePath);

    cout << "FL_pts_all" << endl;
    for (const auto& FL_pts : FL_pts_all) {
        for (const auto& FL_pt : FL_pts) {
            cout << FL_pt << " ";
        }
        cout << endl;
    }

    map<string, Eigen::MatrixXf> colorData = readColorMeasurementData(dataFilePath);
    cout << "Color data" << endl;

    vector<string> color_list = {"red", "green", "blue", "cyan", "magenta", "yellow", "white", "black"};
    vector<Point> points; // Define and initialize this as needed
    vector<vector<int>> colors; // Define and initialize this as needed

    cout << "Generating all possible predicted colors" << endl;
    map<string, map<tuple<int, int, int>, vector<int>>> all_interpolated_FL;

    vector<thread> threads;
    mutex mtx; // Mutex to protect shared data

    for (const auto& color : color_list) {
        vector<tuple<int, int, int>> FL_ranges;
        for (int FL_R = 0; FL_R < 256; FL_R += 5) {
            for (int FL_G = 0; FL_G < 256; FL_G += 5) {
                for (int FL_B = 0; FL_B < 256; FL_B += 5) {
                    FL_ranges.emplace_back(FL_R, FL_G, FL_B);
                }
            }
        }

        threads.emplace_back([&, color, FL_ranges] {
            for (const auto& FL_range : FL_ranges) {
                auto interpolated_colors = interpolate_color_for_color(FL_range, points, colors);
                lock_guard<mutex> lock(mtx);
                all_interpolated_FL[color][FL_range] = interpolated_colors;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}

vector<int> interpolate_color_for_color(tuple<int, int, int> FL_range, const vector<Point>& points, const vector<vector<int>>& colors) {
    // Create a Delaunay triangulation of the input points
    Delaunay_triangulation dt;
    dt.insert(points.begin(), points.end());

    // Convert FL_range to a Point
    Point query_point(get<0>(FL_range), get<1>(FL_range));

    // Perform natural neighbor coordinates
    std::vector<std::pair<Point, K::FT>> coords;
    CGAL::Triple<std::back_insert_iterator<std::vector<std::pair<Point, K::FT>>>, K::FT, bool> result =
        CGAL::natural_neighbor_coordinates_2(dt, query_point, std::back_inserter(coords));

    if (!result.third) {
        // Handle the case where the point is not within the convex hull
        return {};
    }

    // Perform interpolation using the obtained coordinates
    vector<int> interpolated_color(3, 0);
    K::FT sum_weights = result.second;

    for (const auto& coord : coords) {
        Point p = coord.first;
        K::FT weight = coord.second / sum_weights;

        // Find the corresponding color in the original color list
        auto it = find(points.begin(), points.end(), p);
        if (it != points.end()) {
            int index = distance(points.begin(), it);
            for (int i = 0; i < 3; ++i) {
                interpolated_color[i] += weight * colors[index][i];
            }
        }
    }

    return interpolated_color;
}

vector<vector<double>> readFrontLightCombinations(const string& file_path) {
    std::ifstream file(file_path);
    std::vector<double> FL_red, FL_green, FL_blue;
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return {}; // Return an empty vector if file opening fails
    }

    std::string line;
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> rowData;

        // Extract each cell in the row
        while (std::getline(ss, cell, ',')) {
            rowData.push_back(cell);
        }

        // Crop the quotation marks for each cell
        for (auto& cell : rowData) {
            if (!cell.empty() && cell.front() == '"') {
                cell.erase(0, 1);
            }
            if (!cell.empty() && cell.back() == '"') {
                cell.pop_back();
            }
        }

        // Check for enough columns and ignore rows that don't have the expected data
        if (rowData.size() > 3) { // Assuming FL_R, FL_G, and FL_B are the first three valid columns
            double value;
            for (int i = 0; i < 3; ++i) { // Only iterate over the first three values (FL_R, FL_G, FL_B)
                std::istringstream iss(rowData[i]); // Use istringstream for conversion
                
                if (iss >> value) { // Try to read a double value from the string
                    switch(i) {
                        case 0: FL_red.push_back(value); break;
                        case 1: FL_green.push_back(value); break;
                        case 2: FL_blue.push_back(value); break;
                    }
                } else {
                    // Handle the case where conversion fails, e.g., log an error or assign a default value
                    std::cerr << "Conversion failed for value: " << rowData[i] << " in row: " << line << std::endl;
                }
            }
        }
    }

    std::vector<std::vector<double>> FL_pts_all;
    // Assuming the vectors are of equal length
    for (size_t i = 0; i < FL_red.size(); ++i) {
        FL_pts_all.push_back({FL_red[i], FL_green[i], FL_blue[i]});
    }

    return FL_pts_all;
}

// Function to trim quotation marks from a string
std::string trim_quotes(const std::string& str) {
    if (str.size() > 1 && str.front() == '"' && str.back() == '"') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

// Function to read color measurement data from a CSV file
std::map<std::string, Eigen::MatrixXf> readColorMeasurementData(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    std::vector<double> X, Y, Z;

    // Skip the first line (header)
    std::getline(file, line);

    int line_count = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        int cell_count = 0;

        // Skip the first 5 columns
        for (int i = 0; i < 5; ++i) {
            std::getline(ss, cell, ',');
        }

        while (std::getline(ss, cell, ',')) {
            cell = trim_quotes(cell); // Remove quotation marks
            try {
                if (!cell.empty()) {
                    double value = std::stod(cell);

                    int color_index = line_count / 27;
                    int data_index = line_count % 27;

                    switch (color_index) {
                        case 0: X.push_back(value / 255.0); break;
                        case 1: Y.push_back(value / 255.0); break;
                        case 2: Z.push_back(value / 255.0); break;
                    }
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << cell << " at line " << line_count + 1 << ", cell " << cell_count << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << cell << " at line " << line_count + 1 << ", cell " << cell_count << std::endl;
            }
            cell_count++;
        }
        line_count++;
    }

    file.close();

    // Assuming each color has 27 data points
    Eigen::MatrixXf data_pts(X.size(), 3);
    for (size_t i = 0; i < X.size(); ++i) {
        data_pts(i, 0) = X[i];
        data_pts(i, 1) = Y[i];
        data_pts(i, 2) = Z[i];
    }

    // Separate the data points into 8 colors
    std::vector<std::string> color_names = {"red", "green", "blue", "cyan", "magenta", "yellow", "white", "black"};
    std::map<std::string, Eigen::MatrixXf> color_dict;

    for (size_t i = 0; i < color_names.size(); ++i) {
        Eigen::MatrixXf color_pts = data_pts.block<27, 3>(i * 27, 0);
        color_dict[color_names[i]] = color_pts;
    }

    return color_dict;
}