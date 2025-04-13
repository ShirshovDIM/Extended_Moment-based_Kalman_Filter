#include "filter/simple_vehicle_nkf.h"
#include "distribution/two_dimensional_normal_distribution.h"
#include "distribution/three_dimensional_normal_distribution.h"
#include <iostream>
#include <Eigen/Sparse>
#include <chrono>

using namespace SimpleVehicle;

// Функция для аппроксимации итеративным методом Ньютона-Шульца
Eigen::MatrixXd invertNewtonSchulz(const Eigen::MatrixXd& S, int maxIters = 10, double tol = 1e-16) {
    int n = S.rows();
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(n, n);

    // Вычисление начального приближения
    double normS = S.norm(); // Фробениусова норма
    if (normS < 1e-12) {
        throw std::runtime_error("Matrix norm is too small for Newton-Schulz iteration.");
    }
    X = (1.0 / normS) * X;

    for (int iter = 0; iter < maxIters; ++iter) {
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n) - X * S;
        double error = R.norm();
        if (error < tol) {
            break;
        }
        X = X * (Eigen::MatrixXd::Identity(n, n) + R); // Конкретно в нашей задаче эквивалентно X = 2*X - X*S*X
    }
    return X;
}

// Функция для аппроксимации обратной матрицы по алгоритму BFGS
Eigen::MatrixXd invertBFGS(const Eigen::MatrixXd& S, int maxIters = 10, double tol = 1e-6) {
    int n = S.rows();
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n, n);  // начальное приближение: I
    int N2 = n * n;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(N2, N2); // приближение обратной Гессиана в векторном представлении

    // Вычисления градиента f(J) = 0.5 * ||J * S - I||²
    auto computeGrad = [&](const Eigen::MatrixXd& Jcur) -> Eigen::MatrixXd {
        Eigen::MatrixXd E = Jcur * S - Eigen::MatrixXd::Identity(n, n);
        return 2.0 * E * S.transpose();
        };

    auto matToVec = [&](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
        return Eigen::Map<const Eigen::VectorXd>(M.data(), N2);
        };
    auto vecToMat = [&](const Eigen::VectorXd& v) -> Eigen::MatrixXd {
        return Eigen::Map<const Eigen::MatrixXd>(v.data(), n, n);
        };

    Eigen::VectorXd grad = matToVec(computeGrad(J));
    for (int iter = 0; iter < maxIters; ++iter) {
        if (grad.norm() < tol)
            break;
        Eigen::VectorXd p = -H * grad;
        Eigen::MatrixXd Pmat = vecToMat(p);
        Eigen::MatrixXd ES = J * S - Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd PS = Pmat * S;
        double numerator = 2.0 * (ES.array() * PS.array()).sum();
        double denominator = 2.0 * PS.squaredNorm();
        double alpha = (denominator > 1e-12 ? -numerator / denominator : -1.0);
        if (alpha < -1.0) alpha = -1.0;
        if (alpha > 2.0) alpha = 2.0;
        Eigen::VectorXd s_vec = alpha * p;
        Eigen::VectorXd J_new_vec = matToVec(J) + s_vec;
        Eigen::MatrixXd J_new = vecToMat(J_new_vec);
        Eigen::VectorXd grad_new = matToVec(computeGrad(J_new));
        Eigen::VectorXd y_vec = grad_new - grad;
        double sy = s_vec.dot(y_vec);
        if (std::abs(sy) < 1e-12)
            break;
        double rho = 1.0 / sy;
        Eigen::MatrixXd I_N2 = Eigen::MatrixXd::Identity(N2, N2);
        Eigen::MatrixXd Vy = I_N2 - rho * s_vec * y_vec.transpose();
        Eigen::MatrixXd Vz = I_N2 - rho * y_vec * s_vec.transpose();
        H = Vy * H * Vz + rho * (s_vec * s_vec.transpose());
        J = J_new;
        grad = grad_new;
    }
    return J;
}

// Функция для аппроксимации обратной матрицы по алгоритму DFP
Eigen::MatrixXd invertDFP(const Eigen::MatrixXd& S, int maxIters = 10, double tol = 1e-6) {

    int n = S.rows();
    int N2 = n * n;
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(N2, N2);  // приближение Гессиана

    auto computeGrad = [&](const Eigen::MatrixXd& Jcur) -> Eigen::MatrixXd {
        Eigen::MatrixXd E = Jcur * S - Eigen::MatrixXd::Identity(n, n);
        return 2.0 * E * S.transpose();
        };
    auto matToVec = [&](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
        return Eigen::Map<const Eigen::VectorXd>(M.data(), N2);
        };
    auto vecToMat = [&](const Eigen::VectorXd& v) -> Eigen::MatrixXd {
        return Eigen::Map<const Eigen::MatrixXd>(v.data(), n, n);
        };

    Eigen::VectorXd grad = matToVec(computeGrad(J));
    for (int iter = 0; iter < maxIters; ++iter) {
        if (grad.norm() < tol)
            break;
        // Направление поиска p = -H * grad, где H = B⁻¹
        Eigen::MatrixXd H = B.inverse();
        Eigen::VectorXd p = -H * grad;
        Eigen::MatrixXd Pmat = vecToMat(p);
        Eigen::MatrixXd E = J * S - Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd PS = Pmat * S;
        double numerator = 2.0 * (E.array() * PS.array()).sum();
        double denominator = 2.0 * PS.squaredNorm();
        double alpha = (denominator > 1e-12 ? -numerator / denominator : -1.0);
        if (alpha < -1.0) alpha = -1.0;
        if (alpha > 2.0) alpha = 2.0;
        Eigen::VectorXd s_vec = alpha * p;
        Eigen::VectorXd J_new_vec = matToVec(J) + s_vec;
        Eigen::MatrixXd J_new = vecToMat(J_new_vec);
        Eigen::VectorXd grad_new = matToVec(computeGrad(J_new));
        Eigen::VectorXd y_vec = grad_new - grad;
        if (std::abs(s_vec.dot(y_vec)) < 1e-12)
            break;
        double yTs = s_vec.dot(y_vec);
        Eigen::MatrixXd term1 = (y_vec * y_vec.transpose()) / yTs;
        double sBs = s_vec.dot(B * s_vec);
        if (std::abs(sBs) < 1e-12)
            break;
        Eigen::MatrixXd term2 = (B * s_vec * s_vec.transpose() * B) / sBs;
        B = B + term1 - term2;
        J = J_new;
        grad = grad_new;
    }
    return J;
}

// Функция для аппроксимации обратной матрицы по алгоритму L-BFGS
Eigen::MatrixXd invertLBFGS(const Eigen::MatrixXd& S, int maxIters = 10, double tol = 1e-6, int m = 5) {
    int n = S.rows();
    int N2 = n * n;
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n, n);

    auto computeGrad = [&](const Eigen::MatrixXd& Jcur) -> Eigen::MatrixXd {
        Eigen::MatrixXd E = Jcur * S - Eigen::MatrixXd::Identity(n, n);
        return 2.0 * E * S.transpose();
        };
    auto matToVec = [&](const Eigen::MatrixXd& M) -> Eigen::VectorXd {
        return Eigen::Map<const Eigen::VectorXd>(M.data(), N2);
        };
    auto vecToMat = [&](const Eigen::VectorXd& v) -> Eigen::MatrixXd {
        return Eigen::Map<const Eigen::MatrixXd>(v.data(), n, n);
        };

    Eigen::VectorXd grad = matToVec(computeGrad(J));
    std::vector<Eigen::VectorXd> s_history;
    std::vector<Eigen::VectorXd> y_history;
    s_history.reserve(m);
    y_history.reserve(m);

    for (int iter = 0; iter < maxIters; ++iter) {
        if (grad.norm() < tol)
            break;
        // Первый проход: рекурсия L-BFGS
        Eigen::VectorXd q = grad;
        int k = s_history.size();
        std::vector<double> alpha(k);
        for (int i = k - 1; i >= 0; --i) {
            double rho_i = 1.0 / y_history[i].dot(s_history[i]);
            alpha[i] = rho_i * s_history[i].dot(q);
            q = q - alpha[i] * y_history[i];
        }
        double gamma = 1.0;
        if (k > 0) {
            gamma = s_history.back().dot(y_history.back()) / y_history.back().dot(y_history.back());
        }
        Eigen::VectorXd r = gamma * q;
        for (int i = 0; i < k; ++i) {
            double rho_i = 1.0 / y_history[i].dot(s_history[i]);
            double beta = rho_i * y_history[i].dot(r);
            r = r + s_history[i] * (alpha[i] - beta);
        }
        Eigen::VectorXd p = -r;
        Eigen::MatrixXd Pmat = vecToMat(p);
        Eigen::MatrixXd E = J * S - Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd PS = Pmat * S;
        double numerator = 2.0 * (E.array() * PS.array()).sum();
        double denominator = 2.0 * PS.squaredNorm();
        double alpha_step = (denominator > 1e-12 ? -numerator / denominator : -1.0);
        if (alpha_step < -1.0) alpha_step = -1.0;
        if (alpha_step > 2.0) alpha_step = 2.0;
        Eigen::VectorXd s_vec = alpha_step * p;
        Eigen::VectorXd J_new_vec = matToVec(J) + s_vec;
        Eigen::MatrixXd J_new = vecToMat(J_new_vec);
        Eigen::VectorXd grad_new = matToVec(computeGrad(J_new));
        Eigen::VectorXd y_vec = grad_new - grad;
        if (std::abs(s_vec.dot(y_vec)) < 1e-12)
            break;
        if ((int)s_history.size() == m) {
            s_history.erase(s_history.begin());
            y_history.erase(y_history.begin());
        }
        s_history.push_back(s_vec);
        y_history.push_back(y_vec);
        J = J_new;
        grad = grad_new;
    }
    return J;
}

SimpleVehicleNKF::SimpleVehicleNKF() {
    vehicle_model_ = SimpleVehicleModel();
    // Инициализация счетчиков
    diagonal_steps_ = 0;
    sparse_steps_ = 0;
    total_update_steps_ = 0;
}

StateInfo SimpleVehicleNKF::predict(const StateInfo& state_info,
    const Eigen::Vector2d& control_inputs,
    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) {
    // Step1. Approximate to Gaussian Distribution
    const auto state_mean = state_info.mean;
    const auto state_cov = state_info.covariance;
    ThreeDimensionalNormalDistribution dist(state_info.mean, state_info.covariance);

    // Step2. State Moment
    SimpleVehicleModel::StateMoments moment;
    moment.xPow1 = dist.calc_moment(STATE::IDX::X, 1);
    moment.yPow1 = dist.calc_moment(STATE::IDX::Y, 1);
    moment.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    moment.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);
    moment.yawPow1 = dist.calc_moment(STATE::IDX::YAW, 1);
    moment.xPow2 = dist.calc_moment(STATE::IDX::X, 2);
    moment.yPow2 = dist.calc_moment(STATE::IDX::Y, 2);
    moment.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    moment.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    moment.yawPow2 = dist.calc_moment(STATE::IDX::YAW, 2);
    moment.xPow1_yPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::Y);
    moment.cPow1_xPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.sPow1_xPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.cPow1_yPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.sPow1_yPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);
    moment.xPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::X, STATE::IDX::YAW);
    moment.yPow1_yawPow1 = dist.calc_cross_second_moment(STATE::IDX::Y, STATE::IDX::YAW);
    moment.cPow1_yawPow1 = dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1);
    moment.sPow1_yawPow1 = dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1);

    // Step3. Control Input
    SimpleVehicleModel::Controls controls;
    controls.v = control_inputs(INPUT::IDX::V);
    controls.u = control_inputs(INPUT::IDX::U);
    controls.cu = std::cos(controls.u);
    controls.su = std::sin(controls.u);

    // Step4. System Noise
    const auto wv_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WV);
    const auto wu_dist_ptr = noise_map.at(SYSTEM_NOISE::IDX::WU);
    SimpleVehicleModel::SystemNoiseMoments system_noise_moments;
    system_noise_moments.wvPow1 = wv_dist_ptr->calc_moment(1);
    system_noise_moments.wvPow2 = wv_dist_ptr->calc_moment(2);
    system_noise_moments.wuPow1 = wu_dist_ptr->calc_moment(1);
    system_noise_moments.wuPow2 = wu_dist_ptr->calc_moment(2);
    system_noise_moments.cwuPow1 = wu_dist_ptr->calc_cos_moment(1);
    system_noise_moments.swuPow1 = wu_dist_ptr->calc_sin_moment(1);
    system_noise_moments.swuPow2 = wu_dist_ptr->calc_sin_moment(2);
    system_noise_moments.cwuPow2 = wu_dist_ptr->calc_cos_moment(2);
    system_noise_moments.cwuPow1_swuPow1 = wu_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step5. Propagate
    const auto predicted_moment = vehicle_model_.propagateStateMoments(moment, system_noise_moments, controls);

    StateInfo predicted_info;
    predicted_info.mean(STATE::IDX::X) = predicted_moment.xPow1;
    predicted_info.mean(STATE::IDX::Y) = predicted_moment.yPow1;
    predicted_info.mean(STATE::IDX::YAW) = predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::X) = predicted_moment.xPow2 - predicted_moment.xPow1 * predicted_moment.xPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::Y) = predicted_moment.yPow2 - predicted_moment.yPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::YAW) = predicted_moment.yawPow2 - predicted_moment.yawPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y) = predicted_moment.xPow1_yPow1 - predicted_moment.xPow1 * predicted_moment.yPow1;
    predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW) = predicted_moment.xPow1_yawPow1 - predicted_moment.xPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW) = predicted_moment.yPow1_yawPow1 - predicted_moment.yPow1 * predicted_moment.yawPow1;
    predicted_info.covariance(STATE::IDX::Y, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::Y);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::X) = predicted_info.covariance(STATE::IDX::X, STATE::IDX::YAW);
    predicted_info.covariance(STATE::IDX::YAW, STATE::IDX::Y) = predicted_info.covariance(STATE::IDX::Y, STATE::IDX::YAW);

    return predicted_info;
}

StateInfo SimpleVehicleNKF::update(const StateInfo& state_info,
    const Eigen::Vector2d& observed_values,
    const Eigen::Vector2d& landmark,
    const std::map<int, std::shared_ptr<BaseDistribution>>& noise_map) {
    // Измерение времени выполнения
    auto start = std::chrono::high_resolution_clock::now();

    const auto predicted_mean = state_info.mean;
    const auto predicted_cov = state_info.covariance;

    ThreeDimensionalNormalDistribution dist(predicted_mean, predicted_cov);
    SimpleVehicleModel::ReducedStateMoments reduced_moments;
    reduced_moments.cPow1 = dist.calc_cos_moment(STATE::IDX::YAW, 1);
    reduced_moments.sPow1 = dist.calc_sin_moment(STATE::IDX::YAW, 1);
    reduced_moments.cPow2 = dist.calc_cos_moment(STATE::IDX::YAW, 2);
    reduced_moments.sPow2 = dist.calc_sin_moment(STATE::IDX::YAW, 2);
    reduced_moments.xPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1 = dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow1 = dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.cPow1_sPow1 = dist.calc_cos_sin_moment(STATE::IDX::YAW, 1, 1);
    reduced_moments.xPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow2 = dist.calc_x_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_sPow2 = dist.calc_x_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow1_cPow1_sPow1 = dist.calc_x_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow2 = dist.calc_xx_cos_z_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_sPow2 = dist.calc_xx_sin_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    reduced_moments.yPow2_cPow1_sPow1 = dist.calc_xx_cos_z_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    reduced_moments.xPow1_yPow1_cPow2 = dist.calc_xy_cos_z_cos_z_moment();
    reduced_moments.xPow1_yPow1_sPow2 = dist.calc_xy_sin_z_sin_z_moment();
    reduced_moments.xPow1_yPow1_cPow1_sPow1 = dist.calc_xy_cos_z_sin_z_moment();

    // Step2. Create Observation Noise
    const auto wr_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WR);
    const auto wa_dist_ptr = noise_map.at(OBSERVATION_NOISE::IDX::WA);
    SimpleVehicleModel::ObservationNoiseMoments observation_noise;
    observation_noise.wrPow1 = wr_dist_ptr->calc_moment(1);
    observation_noise.wrPow2 = wr_dist_ptr->calc_moment(2);
    observation_noise.cwaPow1 = wa_dist_ptr->calc_cos_moment(1);
    observation_noise.swaPow1 = wa_dist_ptr->calc_sin_moment(1);
    observation_noise.cwaPow2 = wa_dist_ptr->calc_cos_moment(2);
    observation_noise.swaPow2 = wa_dist_ptr->calc_sin_moment(2);
    observation_noise.cwaPow1_swaPow1 = wa_dist_ptr->calc_cos_sin_moment(1, 1);

    // Step3. Get Observation Moments
    const auto observation_moments = vehicle_model_.getObservationMoments(reduced_moments, observation_noise, landmark);

    ObservedInfo observed_info;
    observed_info.mean(OBSERVATION::IDX::RCOS) = observation_moments.rcosPow1;
    observed_info.mean(OBSERVATION::IDX::RSIN) = observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RCOS) = observation_moments.rcosPow2 - observation_moments.rcosPow1 * observation_moments.rcosPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RSIN) = observation_moments.rsinPow2 - observation_moments.rsinPow1 * observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN) = observation_moments.rcosPow1_rsinPow1 - observation_moments.rcosPow1 * observation_moments.rsinPow1;
    observed_info.covariance(OBSERVATION::IDX::RSIN, OBSERVATION::IDX::RCOS) = observed_info.covariance(OBSERVATION::IDX::RCOS, OBSERVATION::IDX::RSIN);

    const auto observation_mean = observed_info.mean;
    const auto observation_cov = observed_info.covariance;

    const double& x_land = landmark(0);
    const double& y_land = landmark(1);
    const double& wrPow1 = observation_noise.wrPow1;
    const double& cwaPow1 = observation_noise.cwaPow1;
    const double& swaPow1 = observation_noise.swaPow1;

    const double xPow1_caPow1 = x_land * dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW)
        - dist.calc_xx_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW)
        + y_land * dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW)
        - dist.calc_xy_sin_z_moment();
    const double xPow1_saPow1 = y_land * dist.calc_x_cos_z_moment(STATE::IDX::X, STATE::IDX::YAW)
        - dist.calc_xy_cos_z_moment()
        - x_land * dist.calc_x_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW)
        + dist.calc_xx_sin_z_moment(STATE::IDX::X, STATE::IDX::YAW);
    const double yPow1_caPow1 = x_land * dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
        - dist.calc_xy_cos_z_moment()
        + y_land * dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
        - dist.calc_xx_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yPow1_saPow1 = y_land * dist.calc_x_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
        - dist.calc_xx_cos_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
        - x_land * dist.calc_x_sin_z_moment(STATE::IDX::Y, STATE::IDX::YAW)
        + dist.calc_xy_sin_z_moment();
    const double yawPow1_caPow1 = x_land * dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1)
        - dist.calc_xy_cos_y_moment(STATE::IDX::X, STATE::IDX::YAW)
        + y_land * dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1)
        - dist.calc_xy_sin_y_moment(STATE::IDX::Y, STATE::IDX::YAW);
    const double yawPow1_saPow1 = y_land * dist.calc_x_cos_x_moment(STATE::IDX::YAW, 1, 1)
        - dist.calc_xy_cos_y_moment(STATE::IDX::Y, STATE::IDX::YAW)
        - x_land * dist.calc_x_sin_x_moment(STATE::IDX::YAW, 1, 1)
        + dist.calc_xy_sin_y_moment(STATE::IDX::X, STATE::IDX::YAW);

    Eigen::MatrixXd state_observation_cov(3, 2); // sigma = E[XY^T] - E[X]E[Y]^T
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RCOS)
        = wrPow1 * cwaPow1 * xPow1_caPow1 - wrPow1 * swaPow1 * xPow1_saPow1
        - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::X, OBSERVATION::IDX::RSIN)
        = wrPow1 * cwaPow1 * xPow1_saPow1 + wrPow1 * swaPow1 * xPow1_caPow1
        - predicted_mean(STATE::IDX::X) * observation_mean(OBSERVATION::IDX::RSIN);
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RCOS)
        = wrPow1 * cwaPow1 * yPow1_caPow1 - wrPow1 * swaPow1 * yPow1_saPow1
        - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::Y, OBSERVATION::IDX::RSIN)
        = wrPow1 * cwaPow1 * yPow1_saPow1 + wrPow1 * swaPow1 * yPow1_caPow1
        - predicted_mean(STATE::IDX::Y) * observation_mean(OBSERVATION::IDX::RSIN);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RCOS)
        = wrPow1 * cwaPow1 * yawPow1_caPow1 - wrPow1 * swaPow1 * yawPow1_saPow1
        - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RCOS);
    state_observation_cov(STATE::IDX::YAW, OBSERVATION::IDX::RSIN)
        = wrPow1 * cwaPow1 * yawPow1_saPow1 + wrPow1 * swaPow1 * yawPow1_caPow1
        - predicted_mean(STATE::IDX::YAW) * observation_mean(OBSERVATION::IDX::RSIN);


    Eigen::MatrixXd S = observation_cov;  // S – матрица наблюдений (covariance)
    Eigen::MatrixXd S_inv;
    Eigen::MatrixXd K(3, 2);
    last_S_ = S;
    ++total_update_steps_; // Увеличиваем счетчик обновлений


    switch (inv_method_) {
    case InversionMethod::NEWTON_SCHULZ:
        S_inv = invertNewtonSchulz(S);
        break;
    case InversionMethod::DIRECT:
        S_inv = S.inverse();
        break;
    case InversionMethod::BFGS:
        S_inv = invertBFGS(S);
        break;
    case InversionMethod::DFP:
        S_inv = invertDFP(S);
        break;
    case InversionMethod::LBFGS:
        S_inv = invertLBFGS(S);
        break;
    default:
        S_inv = S.inverse();
        std::cout << "[NKF] Approximation regime has not been recognized; used DIRECT as default" << std::endl;
    }


    Eigen::MatrixXd K_standard = state_observation_cov * S_inv;


    bool is_diagonal = (std::abs(S(0, 1)) < diagonal_threshold_ && std::abs(S(1, 0)) < diagonal_threshold_);

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    for (int i = 0; i < state_observation_cov.rows(); ++i) {
        for (int j = 0; j < state_observation_cov.cols(); ++j) {
            if (std::abs(state_observation_cov(i, j)) >= sparsity_threshold_) {
                triplet_list.push_back(T(i, j, state_observation_cov(i, j)));
            }
        }
    }
    Eigen::SparseMatrix<double> sparse_state_observation_cov(3, 2);
    sparse_state_observation_cov.setFromTriplets(triplet_list.begin(), triplet_list.end());
    sparse_state_observation_cov.makeCompressed();
    bool is_sparse = (triplet_list.size() < state_observation_cov.size());


    std::cout << "[DEBUG] Sparse state_observation_cov non-zeros:\n";
    for (int k = 0; k < sparse_state_observation_cov.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_state_observation_cov, k); it; ++it) {
            std::cout << " (" << it.row() << "," << it.col() << ") = " << it.value() << std::endl;
        }
    }

    if (is_diagonal && is_sparse) {
        Eigen::Vector2d S_inv_diag;
        S_inv_diag(0) = S_inv(0, 0); 
        S_inv_diag(1) = S_inv(1, 1);

        K.setZero();
        for (int k = 0; k < sparse_state_observation_cov.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_state_observation_cov, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                K(row, col) = it.value() * S_inv_diag(col);
            }
        }

        K = K_standard;

        ++diagonal_steps_;
        ++sparse_steps_;
    }
    else if (is_diagonal) {
        Eigen::Vector2d S_inv_diag;
        S_inv_diag(0) = S_inv(0, 0);
        S_inv_diag(1) = S_inv(1, 1);

        K.setZero();
        for (int i = 0; i < state_observation_cov.rows(); ++i) {
            for (int j = 0; j < state_observation_cov.cols(); ++j) {
                K(i, j) = state_observation_cov(i, j) * S_inv_diag(j);
            }
        }

        K = K_standard;

        ++diagonal_steps_;
    }
    else {
        K = K_standard;
        if (is_sparse) ++sparse_steps_;
    }

    std::cout << "[DEBUG] Kalman Gain K:\n" << K << std::endl;

    // Логирование матриц и ошибки аппроксимации
    last_S_inv_ = S_inv;
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(S.rows(), S.cols());
    last_inv_error_ = (S_inv * S - I_n).norm();

    // Обновление состояния
    StateInfo updated_info;
    updated_info.mean = predicted_mean + K * (observed_values - observation_mean);
    updated_info.covariance = predicted_cov - K * observation_cov * K.transpose();
    std::cout << "[DEBUG] Updated mean:\n" << updated_info.mean << std::endl;
    std::cout << "[DEBUG] Updated covariance:\n" << updated_info.covariance << std::endl;
    std::cout << "[DEBUG] Total updates: " << total_update_steps_
        << ", Diagonal steps: " << diagonal_steps_
        << ", Sparse steps: " << sparse_steps_ << std::endl;
    std::cout << "[DEBUG] Update took " << std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count() << " microseconds" << std::endl;

    return updated_info;
}
