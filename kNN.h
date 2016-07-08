#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>

typedef std::vector<std::vector<double>> matrix;


class KNeighborsClassifier {
private:
    size_t KNeighbors;
    matrix coordinateMatrix;
    size_t coordinateDimentions;
    std::vector<int> labels;
    bool CheckInputFit(const matrix& neighborCoordinates, const std::vector<int>& neighborLabels);
    void CheckInputPredict(const std::vector<double>&) const;
    std::vector<std::pair<int, double>> MeasureDistance(const std::vector<double>&) const;
public:
    KNeighborsClassifier(size_t);
    void Fit(const matrix&, const std::vector<int>&);
    int Predict(const std::vector<double>&) const;
};

