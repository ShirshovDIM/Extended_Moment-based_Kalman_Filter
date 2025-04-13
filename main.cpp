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
        std::ifstream landmark_file("C:/Users/dbezu/Desktop/MKF/Extended_MKF/data/MRCLAM_Dataset1/Landmark_Groundtruth.dat");
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
        std::string odometry_filename = "C:/Users/dbezu/Desktop/MKF/Extended_MKF/data/MRCLAM_Dataset1/Robot"
            + std::to_string(robot_num) + "_Odometry.dat";
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
        std::string ground_truth_filename = "C:/Users/dbezu/Desktop/MKF/Extended_MKF/data/MRCLAM_Dataset1/Robot"
            + std::to_string(robot_num) + "_Groundtruth.dat";
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
        std::string measurement_filename = "C:/Users/dbezu/Desktop/MKF/Extended_MKF/data/MRCLAM_Dataset1/Robot"
            + std::to_string(robot_num) + "_Measurement.dat";
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

    // Внешние возмущения
    const double mean_wv = 0.0;
    const double cov_wv = std::pow(0.1, 2);
    const double mean_wu = 0.0;
    const double cov_wu = std::pow(1.0, 2);

    /////////////////////////////////
    ///// Подготовка логирования и CSV /////
    /////////////////////////////////
    std::string results_dir = "C:/Users/dbezu/Desktop/MKF/Extended_MKF/results_NK_project/";
    std::filesystem::create_directories(results_dir);
    // Создаем CSV файл для метрик
    std::ofstream csvFile(results_dir + "metrics_log.csv");
    csvFile << "Method,TotalTime_ms,AvgInvError,FinalInvError,ApproxMemUsage_KB" << std::endl;

    // Вектор методов обращения для NKF и их имена
    std::vector<SimpleVehicleNKF::InversionMethod> inversionMethods = {
        SimpleVehicleNKF::InversionMethod::DIRECT,
        SimpleVehicleNKF::InversionMethod::BFGS,
        SimpleVehicleNKF::InversionMethod::DFP,
        SimpleVehicleNKF::InversionMethod::LBFGS,
        SimpleVehicleNKF::InversionMethod::NEWTON_SCHULZ
    };
    std::vector<std::string> methodNames = { "DIRECT", "BFGS", "DFP", "LBFGS", "NEWTON_SCHULZ"};

    // Вектора для агрегированных метрик (используем double для всех)
    std::vector<double> totalTimeVec;
    std::vector<double> avgInvErrorVec;
    std::vector<double> finalInvErrorVec;
    std::vector<double> memUsageVec;

    /////////////////////////////////
    ///// Симуляция для NKF с разными методами /////
    /////////////////////////////////
    for (size_t methodIdx = 0; methodIdx < inversionMethods.size(); ++methodIdx) {
        std::cout << "=== Executing experiment for NKF with the following method - " << methodNames[methodIdx] << " ===" << std::endl;

        // Создаем новый экземпляр NKF и устанавливаем выбранный метод обращения
        SimpleVehicleNKF nkf_local;
        nkf_local.setInversionMethod(inversionMethods[methodIdx]);

        // Переинициализируем состояние NKF
        StateInfo nkf_state_info_local;
        nkf_state_info_local.mean = { ground_truth_x.front(), ground_truth_y.front(), ground_truth_yaw.front() };
        nkf_state_info_local.covariance = scenario.ini_cov_;

        // Сброс индексов измерений и ground truth для данного прогона
        size_t measurement_id = 0;
        size_t ground_truth_id = 0;

        // Для сбора метрик по обновлениям
        double sumInvError = 0.0;
        int updateCount = 0;
        std::vector<double> simTimes;    // моменты времени обновлений
        std::vector<double> nkfInvErrors;  // динамика ошибки обращения

        auto startTime = std::chrono::high_resolution_clock::now();

        // Основной цикл симуляции по одометрии (20000 шагов)
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
                        const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                            { SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv * dt, cov_wv * dt * dt) },
                            { SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu * dt, cov_wu * dt * dt) }
                        };
                        nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_map);
                    }
                    // Обновление с измерением
                    const Eigen::Vector2d meas = { measurement_range[measurement_id], measurement_bearing[measurement_id] };
                    const Eigen::Vector2d y = { meas(0) * std::cos(meas(1)), meas(0) * std::sin(meas(1)) };
                    const auto landmark = landmark_map.at(measurement_subject.at(measurement_id));
                    const double updated_dt = std::max(1e-5, dt);
                    const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                        { SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv * updated_dt, cov_wv * updated_dt * updated_dt) },
                        { SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu * updated_dt, cov_wu * updated_dt * updated_dt) }
                    };
                    nkf_state_info_local = nkf_local.update(nkf_state_info_local, y, { landmark.x, landmark.y }, measurement_noise_map);

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
                        const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                            { SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv * dt, cov_wv * dt * dt) },
                            { SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu * dt, cov_wu * dt * dt) }
                        };
                        nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_map);
                    }
                    current_time = ground_truth_time.at(ground_truth_id);
                    ++ground_truth_id;
                }
            }
            // Предсказание до следующего времени
            const double dt = next_time - current_time;
            if (dt > 1e-5) {
                const Eigen::Vector2d inputs = { odometry_v.at(odo_id) * dt, odometry_w.at(odo_id) * dt };
                const std::map<int, std::shared_ptr<BaseDistribution>> system_noise_map = {
                    { SYSTEM_NOISE::IDX::WV, std::make_shared<NormalDistribution>(mean_wv * dt, cov_wv * dt * dt) },
                    { SYSTEM_NOISE::IDX::WU, std::make_shared<NormalDistribution>(mean_wu * dt, cov_wu * dt * dt) }
                };
                nkf_state_info_local = nkf_local.predict(nkf_state_info_local, inputs, system_noise_map);
            }
        } // конец цикла по одометрии

        auto endTime = std::chrono::high_resolution_clock::now();
        long totalTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double avgInvError = (updateCount > 0 ? sumInvError / updateCount : 0.0);
        double finalInvError = (nkfInvErrors.empty() ? 0.0 : nkfInvErrors.back());

        // Оценка использования памяти (приблизительно)
        int n = nkf_local.last_S_.rows();
        double memBytes = 0.0;
        if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::DIRECT) {
            memBytes = n * n * sizeof(double); // n^2 - только исходная матрица - наиболее оптимальный по памяти вариант
        }
        else if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::BFGS ||
            inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::DFP) {
            double N2 = n * n;
            memBytes = N2 * N2 * sizeof(double); // n^4 - матрицы Гессиана
        }
        else if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::LBFGS) {
            double N2 = n * n;
            int m = 5;
            memBytes = 2 * m * N2 * sizeof(double); // 10n² (при m=5) чуть менее оптимально для многомерок, в сравнении с методом Ньютона-Шульца
        }
        else if (inversionMethods[methodIdx] == SimpleVehicleNKF::InversionMethod::NEWTON_SCHULZ) {
            memBytes = 2 * n * n * sizeof(double); // X и R (n x n) - оптимально для многомерных структур
        }
        double memKB = memBytes / 1024.0;

        // Вывод метрик в консоль
        std::cout << "Approximation Method " << methodNames[methodIdx]
            << ": time = " << totalTimeMs << " ms, std err = "
                << avgInvError << ", final err = " << finalInvError
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
            plt::xlabel("Время (с)");
            plt::ylabel("Ошибка обращения (‖J*S - I‖)");
            plt::title("Динамика ошибки для NKF: " + methodNames[methodIdx]);
            plt::legend();
            std::string plotFilename = results_dir + "inv_error_plot_" + methodNames[methodIdx] + ".png";
            plt::save(plotFilename);
            std::cout << "Сохранен график: " << plotFilename << std::endl;
    } // конец цикла по NKF методам

    csvFile.close();

    /////////////////////////////////
    ///// Агрегированные графики по метрикам /////
    /////////////////////////////////
    // Создаем вектор позиций для графиков (ось x)
    std::vector<double> positions;
    for (size_t i = 0; i < methodNames.size(); ++i) {
        positions.push_back(static_cast<double>(i));
    }

    // График: Общее время симуляции (ms)
    plt::figure();
    plt::bar(positions, totalTimeVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Метод");
    plt::ylabel("Общее время (ms)");
    plt::title("Общее время симуляции для NKF");
    plt::save(results_dir + "total_time_bar.png");
    std::cout << "Сохранен график общего времени: " << results_dir + "total_time_bar.png" << std::endl;

    // График: Средняя ошибка обращения
    plt::figure();
    plt::bar(positions, avgInvErrorVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Метод");
    plt::ylabel("Средняя ошибка обращения");
    plt::title("Средняя ошибка обращения для NKF");
    plt::save(results_dir + "avg_inv_error_bar.png");
    std::cout << "Сохранен график средней ошибки: " << results_dir + "avg_inv_error_bar.png" << std::endl;

    // График: Финальная ошибка обращения
    plt::figure();
    plt::bar(positions, finalInvErrorVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Метод");
    plt::ylabel("Финальная ошибка обращения");
    plt::title("Финальная ошибка обращения для NKF");
    plt::save(results_dir + "final_inv_error_bar.png");
    std::cout << "Сохранен график финальной ошибки: " << results_dir + "final_inv_error_bar.png" << std::endl;

    // График: Оценка памяти (KB)
    plt::figure();
    plt::bar(positions, memUsageVec);
    plt::xticks(positions, methodNames);
    plt::xlabel("Метод");
    plt::ylabel("Потребление памяти (KB)");
    plt::title("Оценка памяти для NKF");
    plt::save(results_dir + "mem_usage_bar.png");
    std::cout << "Сохранен график потребления памяти: " << results_dir + "mem_usage_bar.png" << std::endl;

    // Итоговый график траекторий (если данные собраны, этот блок можно доработать отдельно)
    plt::figure_size(1500, 900);
    std::map<std::string, std::string> nkf_keywords;
    std::map<std::string, std::string> ekf_keywords;
    std::map<std::string, std::string> ukf_keywords;
    nkf_keywords.insert(std::pair<std::string, std::string>("label", "NKF"));
    ekf_keywords.insert(std::pair<std::string, std::string>("label", "EKF"));
    ukf_keywords.insert(std::pair<std::string, std::string>("label", "UKF"));
    plt::plot(nkf_x_estimate, nkf_y_estimate, nkf_keywords);
    plt::plot(ekf_x_estimate, ekf_y_estimate, ekf_keywords);
    plt::plot(ukf_x_estimate, ukf_y_estimate, ukf_keywords);
    plt::named_plot("True", x_true_vec, y_true_vec);
    plt::legend();
    plt::title("Траектории");
    std::string finalPlotFilename = "C:/Users/dbezu/Desktop/MKF/Extended_MKF/results_NK/graph.png";
    plt::save(finalPlotFilename);
    plt::show();
    

    return 0;
}

