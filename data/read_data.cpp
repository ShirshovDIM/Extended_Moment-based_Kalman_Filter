#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::string file_path = "MRCLAM_Dataset1/Robot1_Groundtruth.dat";

    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Ошибка: файл не найден! Проверьте путь: " << file_path << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }

    file.close();
    return 0;
}