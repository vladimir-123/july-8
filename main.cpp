#include "kNN.h"

typedef std::vector<std::vector<double>> matrix;

int main(int argc, char const *argv[])
{
	const std::vector<double> firstCoordinate = {3, 4};
	const std::vector<double> secondCoordinate = {1, 2};
	const matrix myMatrix = {firstCoordinate, secondCoordinate};
	const std::vector<int> labels = {0, 1};
	KNeighborsClassifier testKnn(1);
	testKnn.Fit(myMatrix, labels);
	std::cout << testKnn.Predict(std::vector<double> {0, 1}) << "\n";
	return 0;
}