#include "kNN.h"

KNeighborsClassifier::KNeighborsClassifier(size_t kneighbors): KNeighbors(kneighbors) {}

bool KNeighborsClassifier::CheckInputFit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) const {
    if (neighborCoordinates.size() != neighborLabels.size()) {
        std::cerr << "bad input\n" << "neighborCoordinates.size() == " << neighborCoordinates.size() 
                    << " neighborLabels.size() == " << neighborLabels.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < neighborCoordinates.size() - 1; ++i) {
        if (neighborCoordinates[i].size() != neighborCoordinates[i + 1].size()) {
            std::cerr << "Matrix is not rectangular\n" << "neighborCoordinates[i].size() == " << neighborCoordinates[i].size()
                        << " neighborCoordinates[i].size() == " << neighborCoordinates[i + 1].size() << "\n";
            return false;
        }
    }
    return true;
}

void KNeighborsClassifier::Fit(const Matrix& neighborCoordinates, const std::vector<int>& neighborLabels) {
    if (CheckInputFit(neighborCoordinates, neighborLabels)) {
        coordinateMatrix = neighborCoordinates;
        labels = neighborLabels;
        int firstLineInMatrix = 0;
        coordinateDimensions = neighborCoordinates[firstLineInMatrix].size();

        if (coordinateMatrix.size() != labels.size()) {
            std::cerr << "coordinateMatrix.size() != labels.size() is true and it shouldn't be\n";
        }
        if (KNeighbors > coordinateMatrix.size()) {
            std::cerr << "KNeighbors > coordinateMatrix.size() is true and it shouldn't be\n";
        }
    } 
}

void KNeighborsClassifier::CheckInputPredict(const std::vector<double>& objectToPredict) const {
    if (objectToPredict.size() != coordinateDimensions) {
        std::cerr << "vector to predict doesn't match\n" << "objectToPredict.size() == " << objectToPredict.size()
                    << " coordinateDimensions == " << coordinateDimensions << "\n";
    }
}

bool CompareLabelToDistancePair(const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) {
return lhs.second < rhs.second;
}


std::vector<std::pair<int, double>> KNeighborsClassifier::MeasureDistance(const std::vector<double>& objectToPredict) const {
    double distanceSquared;
    std::vector<std::pair<int, double>> output;
    output.reserve(labels.size());
    for (int i = 0; i < coordinateMatrix.size(); ++i) {
        for (int j = 0; j < coordinateDimensions; ++j) {
            distanceSquared += pow(objectToPredict[j] - coordinateMatrix[i][j], 2);
        }
        output.emplace_back(std::make_pair(labels[i], sqrt(distanceSquared)));
        distanceSquared = 0;
    }
    std::sort(output.begin(), output.end(), CompareLabelToDistancePair);
    int border = KNeighbors;
    while (output[border].second == output[border + 1].second) {
        std::cout << "output[border] == " << output[border].second
                    << " output[border + 1]" << output[border + 1].second << "\n";
        ++border;
    }
    output.resize(border);
    return output;
}

bool CompareLabelToFrequencyPair(const std::pair<int, size_t>& lhs, const std::pair<int, size_t>& rhs) {
    return lhs.second > rhs.second;
}

int KNeighborsClassifier::Predict(const std::vector<double>& objectToPredict) const {
    CheckInputPredict(objectToPredict);

    std::vector<std::pair<int, double>> tempVector = MeasureDistance(objectToPredict);
    std::unordered_map<int, size_t> tempMap;
    for (int i = 0; i < tempVector.size(); ++i) {
        ++tempMap[tempVector[i].first];
    }
    std::vector<std::pair<int, size_t>> output;
    output.reserve(tempMap.size());
    for (const auto& x : tempMap) {
        output.emplace_back(x);
    }
    std::sort(output.begin(), output.end(), CompareLabelToFrequencyPair);
    return output.begin()->first;
}