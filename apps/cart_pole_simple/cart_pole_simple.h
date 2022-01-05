#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

// https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
class CartPoleSimple {
 public:
  static constexpr int stateSize = 4;
  // cart parameters
  const double gravity = 9.8;
  const double masscart = 1.0;
  const double masspole = 0.1;
  const double total_mass = (masspole + masscart);
  const double length = 0.5;  // actually half the pole's length
  const double polemass_length = (masspole * length);
  const double force_mag = 30.0;
  const double tau = 0.02;  // seconds between state updates

  // end episode parameters
  const double xThreshold = 2.4;
  const double thetaThreshold = M_PI / 15;
  const int maxSteps = 500;

  unsigned long long numSteps = 0;

 protected:
  // state
  int currStep = 0;
  double x, x_dot, theta, theta_dot;

 public:
  void reset(std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    x = dist(gen);
    x_dot = dist(gen);
    theta = dist(gen);
    theta_dot = dist(gen);
    currStep = 0;
  }

  bool step(const double* action) { return stepPhysics(action[0]); }

  bool step(const float* action) { return stepPhysics(action[0]); }

  std::vector<double> state() const { return {x, x_dot, theta, theta_dot}; }

  double reward() const { return isFailed() ? 0. : 1.; }

 protected:
  bool isFailed() const {
    return std::fabs(x) > xThreshold || std::fabs(theta) > thetaThreshold;
  }

  bool isOver() const { return currStep >= maxSteps || isFailed(); }

  bool stepPhysics(double force) {
    const double costheta = std::cos(theta), sintheta = std::sin(theta);
    const auto temp =
        (force + polemass_length * theta_dot * theta_dot * sintheta) /
        total_mass;
    const auto thetaacc =
        (gravity * sintheta - costheta * temp) /
        (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
    const auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;
    x = x + tau * x_dot;
    x_dot = x_dot + tau * xacc;
    theta = theta + tau * theta_dot;
    theta_dot = theta_dot + tau * thetaacc;

    numSteps++;

    return isOver();
  }
};
