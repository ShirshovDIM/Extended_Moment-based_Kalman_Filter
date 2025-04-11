#ifndef UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H
#define UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Eigen>

#include "model/simple_vehicle_model.h"
#include "distribution/base_distribution.h"

class SimpleVehicleNKF {
public:

    // какой метод использовать для обращения матрицы
    enum class InversionMethod {
        DIRECT,         // Использовать стандартное S.inverse() как делали авторы
        BFGS,           // Аппроксимировать через BFGS
        DFP,            // Аппроксимировать через DFP
        LBFGS,          // Аппроксимировать через L-BFGS
        NEWTON_SCHULZ   //Аппроксимация по Ньютону-Шульцу
    };

    // дефолт метод
    InversionMethod inv_method_ = InversionMethod::DIRECT;

    // Сеттер, чтобы менять метод обращения матрицы из main или других кодов
    void setInversionMethod(InversionMethod method) {
        inv_method_ = method;
    }

    Eigen::MatrixXd last_S_;
    Eigen::MatrixXd last_S_inv_;
    double last_inv_error_ = 0.0;

    SimpleVehicleNKF();

    SimpleVehicle::StateInfo predict(const SimpleVehicle::StateInfo& state_info,
        const Eigen::Vector2d& control_inputs,
        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicle::StateInfo update(const SimpleVehicle::StateInfo& state_info,
        const Eigen::Vector2d& observed_values,
        const Eigen::Vector2d& landmark,
        const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map);

    SimpleVehicleModel vehicle_model_;
};

#endif //UNCERTAINTY_PROPAGATION_SIMPLE_VEHICLE_NKF_H