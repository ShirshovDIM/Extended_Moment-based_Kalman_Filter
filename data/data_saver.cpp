#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Функция для сохранения содержимого файла в CSV
void saveToCSV(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inputFile(inputPath);
    std::ofstream outputFile(outputPath);

    if (!inputFile.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << inputPath << std::endl;
        return;
    }
    if (!outputFile.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл " << outputPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        outputFile << line << "\n";
    }

    inputFile.close();
    outputFile.close();
    std::cout << "Данные сохранены в " << outputPath << std::endl;
}

int main() {
    std::vector<std::string> filesToProcess = {
        "MRCLAM_Dataset1/Robot1_Groundtruth.dat",
        "MRCLAM_Dataset1/Robot1_Measurement.dat",
        "MRCLAM_Dataset1/Robot1_Measurement_updated.dat",
        "MRCLAM_Dataset1/Robot1_Odometry.dat",
        "MRCLAM_Dataset1/Landmark_Groundtruth.dat"
    };

    for (const auto& file : filesToProcess) {
        std::string outputName = file.substr(file.find_last_of("/") + 1);
        outputName.replace(outputName.find(".dat"), 4, ".csv");
        saveToCSV(file, outputName);
    }

    return 0;
}