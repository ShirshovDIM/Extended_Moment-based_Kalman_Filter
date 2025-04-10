#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Eigen>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>

#include "matplotlibcpp.h"
#include "distribution/uniform_distribution.h"
#include "distribution/normal_distribution.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "filter/simple_vehicle_nkf.h"
#include "filter/simple_vehicle_ukf.h"
#include "filter/simple_vehicle_ekf.h"
#include "model/simple_vehicle_model.h"
#include "scenario/simple_vehicle_scenario.h"

using namespace SimpleVehicle;
namespace plt = matplotlibcpp;

struct LandMark {
    LandMark(const double _x, const double _y, const double _std_x, const double _std_y)
        : x(_x), y(_y), std_x(_std_x), std_y(_std_y) {
    }
    double x;
    double y;
    double std_x;
    double std_y;
};

int main() {
    const int robot_num = 2;
    // Создание карты штрих-кодов
    std::map<int, int> barcode_map;
    barcode_map.insert(std::make_pair(23, 5));
    barcode_map.insert(std::make_pair(72, 6));
    barcode_map.insert(std::make_pair(27, 7));
    barcode_map.insert(std::make_pair(54, 8));
    barcode_map.insert(std::make_pair(70, 9));
    barcode_map.insert(std::make_pair(36, 10));
    barcode_map.insert(std::make_pair(18, 11));
    barcode_map.insert(std::make_pair(25, 12));
    barcode_map.insert(std::make_pair(9, 13));
    barcode_map.insert(std::make_pair(81, 14));
    barcode_map.insert(std::make_pair(16, 15));
    barcode_map.insert(std::make_pair(90, 16));
    barcode_map.insert(std::make_pair(61, 17));
    barcode_map.insert(std::make_pair(45, 18));
    barcode_map.insert(std::make_pair(7, 19));
    barcode_map.insert(std::make_pair(63, 20));

    // Чтение файла с информацией по маякам
    std::map<size_t, LandMark> landmark_map;
    {
        std::ifstream landmark_file("D:/Optimization/Extended_MKF-shirshov_dev/data/MRCLAM_Dataset1/Landmark_Groundtruth.dat");
        if (landmark_file.fail()) {
            std::cout << "Failed to Open the landmark truth file" << std::endl;
            return -1;
        }
        size_t id;
        double x, y, std_x, std_y;
        landmark_file >> id >> x >> y >> std_x >> std_y;
        while (!landmark_file.eof()) {
            landmark_map.insert(std::make_pair(id, LandMark(x, y, std_x, std_y)));
            landmark_file >> id >> x >> y >> std_x >> std_y;
        }
        landmark_file.close();
    }

    // Чтение одометрии
    std::vector<double> odometry_time;
    std::vector<double> odometry_v;
    std::vector<double> odometry_w;
    {
        std::string odometry_filename = "D:/Optimization/Extended_MKF-shirshov_dev/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Odometry.dat";
        std::ifstream odometry_file(odometry_filename);
        if (odometry_file.fail()) {
            std::cout << "Failed to Open the odometry file" << std::endl;
            return -1;
        }
        double time, v, w;
        odometry_file >> time >> v >> w;
        while (!odometry_file.eof()) {
            odometry_time.push_back(time);
            odometry_v.push_back(v);
            odometry_w.push_back(w);
            odometry_file >> time >> v >> w;
        }
        odometry_file.close();
    }
    const double base_time = odometry_time.front();
    for (size_t i = 0; i < odometry_time.size(); ++i) {
        odometry_time.at(i) -= base_time;
    }

    // Чтение ground truth
    std::vector<double> ground_truth_time;
    std::vector<double> ground_truth_x;
    std::vector<double> ground_truth_y;
    std::vector<double> ground_truth_yaw;
    {
        std::string ground_truth_filename = "D:/Optimization/Extended_MKF-shirshov_dev/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Groundtruth.dat";
        std::ifstream ground_truth_file(ground_truth_filename);
        if (ground_truth_file.fail()) {
            std::cout << "Failed to Open the ground truth file" << std::endl;
            return -1;
        }
        double time, x, y, yaw;
        ground_truth_file >> time >> x >> y >> yaw;
        while (!ground_truth_file.eof()) {
            if (time - base_time < 0.0) {
                ground_truth_file >> time >> x >> y >> yaw;
                continue;
            }
            ground_truth_time.push_back(time - base_time);
            ground_truth_x.push_back(x);
            ground_truth_y.push_back(y);
            ground_truth_yaw.push_back(yaw);
            ground_truth_file >> time >> x >> y >> yaw;
        }
        ground_truth_file.close();
    }

    // Чтение измерений
    std::vector<double> measurement_time;
    std::vector<size_t> measurement_subject;
    std::vector<double> measurement_range;
    std::vector<double> measurement_bearing;
    {
        std::string measurement_filename = "D:/Optimization/Extended_MKF-shirshov_dev/data/MRCLAM_Dataset1/Robot" + std::to_string(robot_num) + "_Measurement.dat";
        std::ifstream measurement_file(measurement_filename);
        if (measurement_file.fail()) {
            std::cout << "Failed to Open the measurement file" << std::endl;
            return -1;
        }
        double time, range, bearing;
        int id;
        measurement_file >> time >> id >> range >> bearing;
        while (!measurement_file.eof()) {
            if (id == 5 || id == 14 || id == 41 || id == 32 || id == 23 ||
                id == 18 || id == 61 || time - base_time < 0.0) {
                measurement_file >> time >> id >> range >> bearing;
                continue;
            }
            measurement_time.push_back(time - base_time);
            measurement_subject.push_back(barcode_map.at(id));
            measurement_range.push_back(range);
            measurement_bearing.push_back(bearing);
            measurement_file >> time >> id >> range >> bearing;
        }
        measurement_file.close();
    }

    /////////////////////////////////
    ///// Настройка фильтров /////////
    /////////////////////////////////
    SimpleVehicleGaussianScenario scenario;
    // EKF и UKF остаются без изменений
    SimpleVehicleEKF ekf;
    SimpleVehicleUKF ukf;
    // Для NKF будем проводить серию запусков с разными методами
    const auto measurement_noise_map = scenario.observation_noise_map_;

    // Извлекаем дисперсии из measurement_noise_map
    Eigen::Vector2d measurement_noise_variances;
    measurement_noise_variances(0) = measurement_noise_map.at(OBSERVATION_NOISE::IDX::WR)->calc_moment(2); // Дисперсия wr
    measurement_noise_variances(1) = measurement_noise_map.at(OBSERVATION_NOISE::IDX::WA)->calc_moment(2); // Дисперсия wa

    // Внешние возмущения
    const double mean_wv = 0.0;
    const double cov_wv = std::pow(0.1, 2);
    const double mean_wu = 0.0;
    const double cov_wu = std::pow(1.0, 2);

    // Дисперсии для system_noise (wv и wu)
    const Eigen::Vector2d base_system_noise_variances(cov_wv, cov_wu);

    /////////////////////////////////
    ///// Подготовка логирования и CSV /////
    /////////////////////////////////
    std::string results_dir = "D:/Optimization/Extended_MKF-shirshov_dev/results";
    std::filesystem::create_directories(results_dir);
    // Создаем CSV файл для метрик
    std::ofstream csvFile(results_dir + "/metrics_log.csv");
    csvFile << "Method,TotalTime_ms,AvgInvError,FinalInvError,ApproxMemUsage_KB" << std::endl;

    // Вектор методов обращения для NKF и их имена
    std::vector<SimpleVehicleNKF::InversionMethod> inversionMethods = {
        SimpleVehicleNKF::InversionMethod::DIRECT,
        SimpleVehicleNKF::InversionMethod::BFGS,
        SimpleVehicleNKF::InversionMethod::DFP,
        SimpleVehicleNKF::InversionMethod::LBFGS
    };
    std::vector<std::string> methodNames = { "DIRECT", "BFGS", "DFP", "LBFGS" };

    // Вектора для агрегированных метрик (используем double для всех)
    std::vector<double> totalTimeVec;
    std::vector<double> avgInvErrorVec;
    std::vector<double> finalInvErrorVec;
    std::vector<double> memUsageVec;

    /////////////////////////////////
    ///// Симуляция для NKF с разными методами /////
    /////////////////////////////////
    for (size_t methodIdx = 0; methodIdx < inversionMethods.size(); ++methodIdx) {
        std::cout << "=== Launching NKF with the method " << methodNames[methodIdx] << " ===" << std::endl;

        SimpleVehicleNKF nkf_local;
        nkf_local.setInversionMethod(inversionMethods[methodIdx]);

        // Переинициализируем состояние NKF
        StateInfo nkf_state_info_local;
        // Явно инициализируем mean как Eigen::Vector3d
        nkf_state_info_local.mean = Eigen::Vector3d(ground_truth_x.front(), ground_truth_y.front(), ground_truth_yaw.front());
        // Инициализируем covariance
        auto ini_cov = scenario.ini_cov_;
        // Убедимся, что матрица симметрична
        ini_cov = (ini_cov + ini_cov.transpose()) / 2.0;
        // Проверяем на NaN и инициализируем нули, если нужно
        if (ini_cov.hasNaN()) {
            std::cout << "[Main] Initial covariance contains NaN, resetting to identity matrix" << std::endl;
            ini_cov = Eigen::Matrix3d::Identity();
        }
        // Проверяем собственные значения и корректируем, если нужно
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(ini_cov);
        if (eigen_solver.eigenvalues().minCoeff() <= 0) {
            const double epsilon = 1e-6;
            ini_cov += epsilon * Eigen::MatrixXd::Identity(ini_cov.rows(), ini_cov.cols());
            std::cout << "[Main] Initial covariance was not positive definite, corrected with epsilon = " << epsilon << std::endl;
        }
        nkf_state_info_local.covariance = ini_cov;

        // Сброс индексов измерений и ground truth для данного прогона
        size_t measurement_id = 0;
        size_t ground_truth_id = 0;

        // Для сбора метрик по обновлениям
        double sumInvError = 0.0;
        int updateCount = 0;
        std::vector<double> simTimes;    // моменты времени обновлений
        std::vector<double> nkfInvErrors;  // динамика ошибки обращения

        auto startTime = std::chrono::high_resolution_clock::now();

        
        for (size_t odo_id = 0; odo_id < 20000; ++odo_id) {
            double current_time = odometry_time.at(odo_id);
            const double next_time = odometry_time.at(odo_id + 1);

            // Обработка измерений
            if (measurement_id < measurement_time.size() && next_time - measurement_time.at(measurement_id) > 0.0) {
                while (measurement_id < measurement_time.size()) {
                    if (next_time - measurement_time.at(measurement_id) < 0.0)
                        break;
                    const double dt = measurement_time.at(measurement_id) - current_time;
                    // Предсказание до измерения
                    if (dt > 1e-5) {
                        const Eigen::Vector2d inputs = { odometry_v.at(odo_id) * dt, odometry_w.at(odo_id) * dt };
                        // Масштабируем дисперсии с учётом dt
                        Eigen::Vector2d system_noise_variances = base_system_noise_variances * (dt * dt);
                        nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_variances);
                    }
                    // Обновление с измерением
                    const Eigen::Vector2d meas = { measurement_range[measurement_id], measurement_bearing[measurement_id] };
                    const Eigen::Vector2d y = { meas(0) * std::cos(meas(1)), meas(0) * std::sin(meas(1)) };
                    const auto landmark = landmark_map.at(measurement_subject.at(measurement_id));
                    const double updated_dt = std::max(1e-5, dt);
                    // Масштабируем дисперсии с учётом updated_dt
                    Eigen::Vector2d system_noise_variances = base_system_noise_variances * (updated_dt * updated_dt);
                    nkf_state_info_local = nkf_local.update(nkf_state_info_local, y, { landmark.x, landmark.y }, measurement_noise_variances);

                    current_time = measurement_time.at(measurement_id);
                    ++measurement_id;

                    // Сохраняем метрику текущего обновления: ошибка обращения
                    double curInvError = nkf_local.last_inv_error_;
                    sumInvError += curInvError;
                    ++updateCount;
                    simTimes.push_back(current_time);
                    nkfInvErrors.push_back(curInvError);
                }
                if (measurement_id == measurement_time.size())
                    break;
            }

            // Предсказание до ground truth (упрощенно)
            while (ground_truth_time.at(ground_truth_id) < current_time && ground_truth_time.at(ground_truth_id) < next_time)
                ++ground_truth_id;
            if (current_time < ground_truth_time.at(ground_truth_id) && ground_truth_time.at(ground_truth_id) < next_time) {
                while (true) {
                    if (next_time < ground_truth_time.at(ground_truth_id))
                        break;
                    const double dt = ground_truth_time.at(ground_truth_id) - current_time;
                    if (dt > 1e-5) {
                        const Eigen::Vector2d inputs = { odometry_v.at(odo_id) * dt, odometry_w.at(odo_id) * dt };
                        // Масштабируем дисперсии с учётом dt
                        Eigen::Vector2d system_noise_variances = base_system_noise_variances * (dt * dt);
                        nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_variances);
                    }
                    current_time = ground_truth_time.at(ground_truth_id);
                    ++ground_truth_id;
                }
            }
            // Предсказание до следующего времени
            const double dt = next_time - current_time;
            if (dt > 1e-5) {
                const Eigen::Vector2d inputs = { odometry_v.at(odo_id) * dt, odometry_w.at(odo_id) * dt };
                // Масштабируем дисперсии с учётом dt
                Eigen::Vector2d system_noise_variances = base_system_noise_variances * (dt * dt);
                nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_variances);
            }
        } // конец цикла по одометрии

        auto endTime = std::chrono::high_resolution_clock::now();
        long totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double avgInvError = (updateCount > 0 ? sumInvError / updateCount : 0.0);
        double finalInvError = (nkfInvErrors.empty() ? 0.0 : nkfInvErrors.back());

        // Оценка использования памяти (приблизительно)
        int n = 2; // Размер матрицы S (2x2, так как observation_cov — 2x2)
        double memBytes = 0.0;
        if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::DIRECT) {
            memBytes = n * n * sizeof(double);
        }
        else if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::BFGS ||
            inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::DFP) {
            double N2 = n * n;
            memBytes = N2 * N2 * sizeof(double);
        }
        else if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::LBFGS) {
            double N2 = n * n;
            int m = 5;
            memBytes = 2 * m * N2 * sizeof(double);
        }
        memBytes += 2 * n * n * sizeof(double) + sizeof(double); 
        double memKB = memBytes / 1024.0;

        // Вывод метрик в консоль
        std::cout << "Method " << methodNames[methodIdx]
            << ": time = " << totalTimeMs << " ms"
                << ", mean error = " << avgInvError
                << ", final error = " << finalInvError
                << ", memory usage = " << memKB << " KB" << std::endl;

            // Запись метрик в CSV файл
            csvFile << methodNames[methodIdx] << ","
                << totalTimeMs << ","
                << avgInvError << ","
                << finalInvError << ","
                << memKB << std::endl;

            // Сохраняем агрегированные метрики
            totalTimeVec.push_back(static_cast<double>(totalTimeMs));
            avgInvErrorVec.push_back(avgInvError);
            finalInvErrorVec.push_back(finalInvError);
            memUsageVec.push_back(memKB);

            // Построение графика динамики ошибки для данного метода NKF
            plt::figure();
            plt::plot(simTimes, nkfInvErrors, { {"label", methodNames[methodIdx]} });
            plt::xlabel("Time (sec)");
            plt::ylabel("Conversion error (‖J*S - I‖)");
            plt::title("Error dynamics for NKF: " + methodNames[methodIdx]);
            plt::legend();
            std::string plotFilename = results_dir + "/inv_error_plot_" + methodNames[methodIdx] + ".png";
            plt::save(plotFilename);
            std::cout << "Plot saved: " << plotFilename << std::endl;
    } // конец цикла по NKF методам

    csvFile.close();


    std::vector<double> positions;
    for (size_t i = 0; i < methodNames.size(); ++i) {
        positions.push_back(static_cast<double>(i));
    }

    // График: Общее время симуляции (ms)
    plt::figure();
    plt::bar(positions, totalTimeVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Method");
    plt::ylabel("Total time (ms)");
    plt::title("Total simulation time for NKF");
    plt::save(results_dir + "/total_time_bar.png");
    std::cout << "Total time schedule saved: " << results_dir + "/total_time_bar.png" << std::endl;

    // График: Средняя ошибка обращения
    plt::figure();
    plt::bar(positions, avgInvErrorVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Method");
    plt::ylabel("Average conversion error");
    plt::title("Average conversion error for NKF");
    plt::save(results_dir + "/avg_inv_error_bar.png");
    std::cout << "Avg error plot saved: " << results_dir + "/avg_inv_error_bar.png" << std::endl;

    // График: Финальная ошибка обращения
    plt::figure();
    plt::bar(positions, finalInvErrorVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Method");
    plt::ylabel("Final error of conversion");
    plt::title("Final error of conversion for NKF");
    plt::save(results_dir + "/final_inv_error_bar.png");
    std::cout << "Final error conversion plot saved: " << results_dir + "/final_inv_error_bar.png" << std::endl;

    // График: Оценка памяти (KB)
    plt::figure();
    plt::bar(positions, memUsageVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Method");
    plt::ylabel("Memory usage (KB)");
    plt::title("Memory usage for NKF");
    plt::save(results_dir + "/mem_usage_bar.png");
    std::cout << "Memory usage plot saved: " << results_dir + "/mem_usage_bar.png" << std::endl;

    return 0;
}