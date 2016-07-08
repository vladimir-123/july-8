#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>

typedef std::vector<std::vector<double>> Matrix;


class KNeighborsClassifier {
private:
    size_t KNeighbors;
    Matrix coordinateMatrix;
    size_t coordinateDimensions;
    std::vector<int> labels;
    bool CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const;
    void CheckInputPredict(const std::vector<double>&) const;
    std::vector<std::pair<int, double>> MeasureDistance(const std::vector<double>&) const;
public:
    KNeighborsClassifier(size_t);
    void Fit(const Matrix&, const std::vector<int>&);
    int Predict(const std::vector<double>&) const;
};

