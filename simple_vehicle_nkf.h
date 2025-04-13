#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>
#include <Eigen/Sparse>


#include "model/simple_vehicle_model.h"
#include "distribution/base_distribution.h"

class SimpleVehicleNKF {
public:
    // Какой метод использовать для обращения матрицы
    enum class InversionMethod {
        DIRECT,         // Использовать стандартное S.inverse() как делали авторы
        BFGS,           // Аппроксимировать через BFGS
        DFP,            // Аппроксимировать через DFP
        LBFGS,          // Аппроксимировать через L-BFGS
        NEWTON_SCHULZ   // Аппроксимация по Ньютону-Шульцу
    };

    SimpleVehicleNKF();

    // Сеттер для метода обращения матрицы

    void setInversionMethod(InversionMethod method) {
        inv_method_ = method;
    }


    void setDiagonalThreshold(double threshold) { diagonal_threshold_ = threshold; }
    void setSparsityThreshold(double threshold) { sparsity_threshold_ = threshold; }
    void setUseSparseOptimization(bool use) { use_sparse_optimization_ = use; }

    // Геттеры для доступа к статистике оптимизации
    int getDiagonalSteps() const { return diagonal_steps_; }
    int getSparseSteps() const { return sparse_steps_; }
    int getTotalUpdateSteps() const { return total_update_steps_; }

    SimpleVehicle::StateInfo predict(const SimpleVehicle::StateInfo& state_info,
        const Eigen::Vector2d& control_inputs,
        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicle::StateInfo update(const SimpleVehicle::StateInfo& state_info,
        const Eigen::Vector2d& observed_values,
        const Eigen::Vector2d& landmark,
        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    // Публичные переменные для логирования (оставлены как в исходном коде)
    Eigen::MatrixXd last_S_;
    Eigen::MatrixXd last_S_inv_;
    double last_inv_error_ = 0.0;

private:
    InversionMethod inv_method_ = InversionMethod::DIRECT;

    // Параметры для оптимизации
    double diagonal_threshold_ = 1e-6; // Порог для диагональности S
    double sparsity_threshold_ = 1e-6; // Порог для разреженности state_observation_cov
    bool use_sparse_optimization_ = true;

    // Счетчики для статистики
    int diagonal_steps_ = 0; // Количество шагов с диагональной S
    int sparse_steps_ = 0;   // Количество шагов с разреженной state_observation_cov
    int total_update_steps_ = 0; // Общее количество обновлений

    SimpleVehicleModel vehicle_model_;
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H
