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

// CGAL Kernel`
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
typedef K::Point_2 Point;
typedef std::map<Point, K::FT> Coord_map;

vector<int> interpolate_color_for_color(tuple<int, int, int> FL_range, const vector<Point>& points, const vector<vector<int>>& colors);
vector<vector<double>> readFrontLightCombinations(const string& file_path);
map<string, Eigen::MatrixXf> readColorMeasurementData(const string& filePath);

int main() {
    string dataFilePath = "../input/data/i1_8colors_27FL_v1.csv";
    
    // Read the color measurement data from the file and process Delaunay triangulation to interpolate colors
    map<string, Eigen::MatrixXf> colorData = readColorMeasurementData(dataFilePath);
    auto FL_pts_all = readFrontLightCombinations(dataFilePath);

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
                    if (FL_R == 255 || FL_G == 255 || FL_B == 255) {
                        FL_ranges.emplace_back(FL_R, FL_G, FL_B);
                    }
                }
            }
        }

        vector<future<vector<int>>> futures;
        for (const auto& FL_range : FL_ranges) {
            futures.emplace_back(async(launch::async, interpolate_color_for_color, FL_range, ref(points), ref(colors)));
        }

        map<tuple<int, int, int>, vector<int>> interpolated;
        for (size_t i = 0; i < FL_ranges.size(); ++i) {
            interpolated[FL_ranges[i]] = futures[i].get();
        }

        lock_guard<mutex> lock(mtx);
        all_interpolated_FL[color] = move(interpolated);
    }

    return 0;
}

// Function to interpolate color for a given FL_range
vector<int> interpolate_color_for_color(tuple<int, int, int> FL_range, const vector<Point>& points, const vector<vector<int>>& colors) {
    Point query_point(get<0>(FL_range), get<1>(FL_range));

    // Setup Delaunay triangulation and insert points
    Delaunay_triangulation dt;
    Coord_map measured_colors_R, measured_colors_G, measured_colors_B;
    for (size_t i = 0; i < points.size(); ++i) {
        dt.insert(points[i]);
        measured_colors_R[points[i]] = colors[i][0];
        measured_colors_G[points[i]] = colors[i][1];
        measured_colors_B[points[i]] = colors[i][2];
    }

    // Helper lambda to interpolate a single channel
    auto interpolate_channel = [&](Coord_map& measured_colors) -> int {
        Coord_map::const_iterator it = measured_colors.find(query_point);
        if (it != measured_colors.end()) {
            return static_cast<int>(it->second);
        } else {
            vector<pair<Point, K::FT>> coords;
            K::FT norm;
            CGAL::natural_neighbor_coordinates_2(dt, query_point, back_inserter(coords), norm);
            K::FT result = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Identity_property_map<K::FT>());
            return static_cast<int>(result);
        }
    };

    // Interpolation
    vector<int> interpolated_color(3, 0); // RGB
    interpolated_color[0] = interpolate_channel(measured_colors_R);
    interpolated_color[1] = interpolate_channel(measured_colors_G);
    interpolated_color[2] = interpolate_channel(measured_colors_B);

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

map<string, Eigen::MatrixXf> readColorMeasurementData(const string& filePath) {
    // Implement reading the color measurement data from the Excel file
}