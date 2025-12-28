#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <sofa.h>

// -----------------------------------------------------------------------------
// Физические и численные константы 
// -----------------------------------------------------------------------------
const double G = 6.67430e-11;                 // Ньютоновская гравитационная постоянная 
const double PI = 3.14159265358979323846;     // π
const double C = 299792458.0;                 // Скорость света в вакууме 
const double SUN_MASS = 1.989e30;             // Масса Солнца 
const double AU = 149597870700.0;             // Астрономическая единица 
const double DEG_TO_RAD = PI / 180.0;         // Градусы → радианы
const double RAD_TO_DEG = 180.0 / PI;         // Радианы → градусы
const double ARCSEC_TO_RAD = PI / (180.0 * 3600.0); // Угловая секунда → радианы
const double RAD_TO_ARCSEC = 180.0 * 3600.0 / PI;  // Радианы → угл.сек
const double JUPITER_MASS = 1.898e27;         // Масса Юпитера 

// -----------------------------------------------------------------------------
// Флаги управления релятивистскими поправками и поведением якобианов
// -----------------------------------------------------------------------------
const bool APPLY_GRAVITATIONAL_DEFLECTION = false; // Учитывать ли искривление луча светом (Солнце)
const bool APPLY_ABERRATION = false;               // Учитывать ли релятивистскую аберрацию
const bool APPLY_LIGHT_TIME = true;                // Учитывать ли световое время (задержку распространения света)
const bool FIX_LIGHT_TIME_IN_JACOBIAN = true;      // Фиксировать эффект светового времени при дифференцировании
const bool FIX_GRAV_DEFLECTION_IN_JACOBIAN = true; // Фиксировать гравитационное отклонение в якобиане

// -----------------------------------------------------------------------------
// Параметры Земли и интеграции
// -----------------------------------------------------------------------------
const double EARTH_RADIUS = 6378137.0;    // Экваториальный радиус Земли 
const double EARTH_OMEGA = 7.2921150e-5;  // Угловая скорость вращения Земли

// Параметры итерационного метода (Гаусс-Ньютона / Ньютона)
const int MAX_ITERATIONS = 5;                   // Максимальное число итераций обратной задачи
const double CONVERGENCE_TOLERANCE = 1e-10;     // Критерий сходимости по норме поправки [m]
const double DAMPING_FACTOR = 1.0;              // Коэффициент демпфирования шага (1.0 = без демпфирования)
const double MAX_STEP_NORM = 1e7;               // Максимальная норма шага Δβ 

// Регуляризация 
const double TIKHONOV_LAMBDA = 1e-10;  
const double MAX_CONDITION_NUMBER = 1e12;

// -----------------------------------------------------------------------------
// Простые типы: 3D-вектор и 3x3 матрица с удобными операциями
// Комментарии объясняют поведение операций и единицы измерения (м и м/с)
// -----------------------------------------------------------------------------
struct Vec3d {
    double x, y, z;

    // Конструкторы
    Vec3d() : x(0), y(0), z(0) {}
    Vec3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Арифметика векторов (компонентно)
    Vec3d operator+(const Vec3d& other) const { return Vec3d(x + other.x, y + other.y, z + other.z); }
    Vec3d operator-(const Vec3d& other) const { return Vec3d(x - other.x, y - other.y, z - other.z); }
    Vec3d operator*(double scalar) const { return Vec3d(x * scalar, y * scalar, z * scalar); }
    Vec3d operator/(double scalar) const { return Vec3d(x / scalar, y / scalar, z / scalar); }

    // Скалярное произведение (вспомогательное)
    double operator*(const Vec3d& other) const { return x * other.x + y * other.y + z * other.z; }

    // Векторное произведение (правило правой руки)
    Vec3d cross(const Vec3d& other) const {
        return Vec3d(y * other.z - z * other.y,
                     z * other.x - x * other.z,
                     x * other.y - y * other.x);
    }

    // Норма и квадрат нормы (удобно для оптимизации)
    double length() const { return std::sqrt(x*x + y*y + z*z); }
    double length_squared() const { return x*x + y*y + z*z; }

    // Возвращает нормализованный вектор; если длина ~0, возвращаем исходный вектор
    Vec3d normalized() const {
        double len = length();
        if (len < 1e-15) return *this;
        return *this / len;
    }

    // Преобразование в Eigen::Vector3d для использования линейной алгебры
    Eigen::Vector3d toEigen() const { return Eigen::Vector3d(x, y, z); }
    static Vec3d fromEigen(const Eigen::Vector3d& v) { return Vec3d(v(0), v(1), v(2)); }

    // Отладочная печать с хорошей точностью
    void print(const std::string& name) const {
        std::cout << std::setprecision(12) << name << ": (" << x << ", " << y << ", " << z << ")" << std::endl;
    }
};

// Простая 3x3 матрица (используется для якобианов и матриц чувствительности)
struct Mat3x3 {
    double data[3][3];

    Mat3x3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                data[i][j] = (i == j) ? 1.0 : 0.0;
    }

    // Сложение, вычитание и умножение на скаляр 
    Mat3x3 operator+(const Mat3x3& other) const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Mat3x3 operator-(const Mat3x3& other) const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    Mat3x3 operator*(double scalar) const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }

    // Умножение матрицы на вектор 
    Vec3d operator*(const Vec3d& v) const {
        return Vec3d(
            data[0][0] * v.x + data[0][1] * v.y + data[0][2] * v.z,
            data[1][0] * v.x + data[1][1] * v.y + data[1][2] * v.z,
            data[2][0] * v.x + data[2][1] * v.y + data[2][2] * v.z
        );
    }

    // Классическое умножение матриц
    Mat3x3 operator*(const Mat3x3& other) const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.data[i][j] = 0.0;
                for (int k = 0; k < 3; ++k)
                    result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
        return result;
    }

    // Транспонирование 
    Mat3x3 transpose() const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.data[i][j] = data[j][i];
        return result;
    }

    // Преобразование в Eigen::Matrix3d и обратно — полезно для численных решателей
    Eigen::Matrix3d toEigen() const {
        Eigen::Matrix3d m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = data[i][j];
        return m;
    }

    static Mat3x3 fromEigen(const Eigen::Matrix3d& m) {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result.data[i][j] = m(i, j);
        return result;
    }
};

// -----------------------------------------------------------------------------
// Описание небесного тела и наблюдения
// -----------------------------------------------------------------------------
struct Object {
    Vec3d position;   // Положение в декартовой системе [m]
    Vec3d velocity;   // Скорость [m/s]
    double mass;      // Масса 
    std::string name; // Читаемое имя 
    int id;           // Простейший идентификатор
    bool is_variable; // Флаг — является ли объект параметром, подлежащим варьированию

    Object() : mass(0), name(""), id(-1), is_variable(false) {}
    Object(const std::string& n, double m, int i, bool var = false) :
        mass(m), name(n), id(i), is_variable(var) {}
};

// Наблюдение: хранит времена, наблюдаемые углы и координаты обсерватории
struct Observation {
    double time_jd;            // Время наблюдения в JD (обычно TDB)
    double time_seconds;       // То же время в секундах от J2000
    double ra_observed;        // Наблюденное прямое восхождение [рад]
    double dec_observed;       // Наблюденное склонение [рад]
    Vec3d obs_position_gcrs;   // Положение обсерватории в GCRS [m]
    Vec3d obs_position_bcrs;   // (опционально) положение в BCRS — заполняется в коде
    Vec3d obs_velocity_gcrs;   // Скорость обсерватории в GCRS [m/s]
    Vec3d obs_velocity_bcrs;   // Скорость в BCRS — заполняется в процессе
};

// PV: пара положение/скорость, используется при преобразованиях систем координат
struct PV {
    Vec3d r; // position [m]
    Vec3d v; // velocity [m/s]
};

// -----------------------------------------------------------------------------
// Прототипы функций 
// -----------------------------------------------------------------------------
Vec3d apply_gravitational_deflection(const Vec3d& direction_unit, const Vec3d& obs_bary,
                                     const Vec3d& ast_bary, const Vec3d& sun_bary);
Vec3d apply_aberration(const Vec3d& direction_unit, const Vec3d& observer_velocity_bcrs);

double compute_light_time_correction(double t_obs, const Vec3d& obs_bary,
                                     const class CubicHermiteInterpolator& ast_interp,
                                     double tolerance = 1e-9, int max_iter = 20);

// -----------------------------------------------------------------------------
// Вспомогательные функции времени: перевод JD <-> секунды от J2000
// -----------------------------------------------------------------------------
double jd_to_seconds(double jd) {
    const double j2000_jd = 2451545.0; // JD для эпохи J2000.0
    return (jd - j2000_jd) * 86400.0;  // Перевод в секунды
}

double seconds_to_jd(double seconds) {
    const double j2000_jd = 2451545.0;
    return j2000_jd + seconds / 86400.0;
}

// -----------------------------------------------------------------------------
// Явная формула якобиана ускорения тела по его относительному положению:
// ∂a/∂r = G*m * (3 r⊗r / |r|^5 - I / |r|^3)
// Здесь r = r_body - r_asteroid (вектор от астероида к телу)
// -----------------------------------------------------------------------------
Mat3x3 compute_acceleration_jacobian(const Vec3d& r_asteroid, const Vec3d& r_body, double mass_body) {
    Mat3x3 J;

    // Вектор от астероида к возмущающему телу
    Vec3d r = r_body - r_asteroid;
    double dist2 = r * r;
    double dist = std::sqrt(dist2);

    // Защита от деления на ноль: если расстояние ~0, возвращаем нулевой вклад
    if (dist < 1e-10) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                J.data[i][j] = 0.0;
        return J;
    }

    double dist3 = dist2 * dist;
    double dist5 = dist2 * dist3;
    double G_m = G * mass_body;

    // Заполнение по формуле: компоненты тензора
    double prefactor = G_m / dist5;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double ri = (i == 0) ? r.x : (i == 1) ? r.y : r.z;
            double rj = (j == 0) ? r.x : (j == 1) ? r.y : r.z;
            J.data[i][j] = prefactor * (3.0 * ri * rj);
            if (i == j) J.data[i][j] -= G_m / dist3; // вычитание G*m/r^3 для диагонали
        }
    }

    return J;
}

// -----------------------------------------------------------------------------
// Вычисление ускорений для всех тел системы и суммарного якобиана ускорения
// для астероида (тот, по которому будем дифференцировать систему)
// positions: вектора положений всех тел
// -----------------------------------------------------------------------------
std::pair<std::vector<Vec3d>, Mat3x3> compute_accelerations_and_jacobian(
    const std::vector<Vec3d>& positions,
    const std::vector<Object>& objects,
    int asteroid_index) {

    size_t n = objects.size();
    std::vector<Vec3d> accelerations(n); // результат ускорений для каждого тела
    Mat3x3 jacobian_sum;                  // суммируем вклад всех тел в якобиан астероида

    // Инициализация нулевого якобиана
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            jacobian_sum.data[i][j] = 0.0;

    // Для каждого тела считаем вклады от остальных тел
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue; // не учитываем самодействие

            Vec3d r = positions[j] - positions[i]; // вектор из i в j
            double dist2 = r * r;
            double dist = std::sqrt(dist2);

            if (dist < 1e-10) continue; // избегаем деления на ноль

            Vec3d dir = r / dist; // единичный вектор
            accelerations[i] = accelerations[i] + (dir * (G * objects[j].mass / dist2)); // a = G m / r^2 * dir

            // Если рассматриваем астероид, суммируем якобиан ∂a/∂r для него
            if (i == asteroid_index) {
                Mat3x3 J = compute_acceleration_jacobian(positions[i], positions[j], objects[j].mass);
                jacobian_sum = jacobian_sum + J;
            }
        }
    }

    return {accelerations, jacobian_sum};
}

// -----------------------------------------------------------------------------
// Расширённая структура состояния для интеграции уравнений вариаций:
// - обычные координаты и скорости всех тел
// - матрицы чувствительности dr/dβ и dv/dβ (3x3) для параметра β (начальное положение астероида)
// - текущее время интегрирования
// Объявлены операторы для удобной работы в методе Рунге-Кутты (сложение, умножение на скаляр)
// -----------------------------------------------------------------------------
struct ExtendedSystemState {
    std::vector<Vec3d> positions;  // положения всех тел [m]
    std::vector<Vec3d> velocities; // скорости всех тел [m/s]
    Mat3x3 dr_dbeta;               // ∂r/∂β — матрица чувствительности положения астероида
    Mat3x3 dv_dbeta;               // ∂v/∂β — матрица чувствительности скорости астероида
    double time = 0.0;             // текущее время интегрирования [s]

    // Побочные операции для Runge-Kutta: складываем состояния и умножаем на скаляр
    ExtendedSystemState operator+(const ExtendedSystemState& other) const {
        ExtendedSystemState result;
        result.positions.resize(positions.size());
        result.velocities.resize(velocities.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            result.positions[i] = positions[i] + other.positions[i];
            result.velocities[i] = velocities[i] + other.velocities[i];
        }

        result.dr_dbeta = dr_dbeta + other.dr_dbeta;
        result.dv_dbeta = dv_dbeta + other.dv_dbeta;

        return result;
    }

    ExtendedSystemState operator*(double scalar) const {
        ExtendedSystemState result;
        result.positions.resize(positions.size());
        result.velocities.resize(velocities.size());

        for (size_t i = 0; i < positions.size(); ++i) {
            result.positions[i] = positions[i] * scalar;
            result.velocities[i] = velocities[i] * scalar;
        }

        result.dr_dbeta = dr_dbeta * scalar;
        result.dv_dbeta = dv_dbeta * scalar;

        return result;
    }

    // Быстрая очистка/инициализация состояния — оставляем dr_dbeta = I, dv_dbeta = 0 по умолчанию
    void clear() {
        positions.clear();
        velocities.clear();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                dr_dbeta.data[i][j] = (i == j) ? 1.0 : 0.0; // по умолчанию: зависимость положения от β = I
                dv_dbeta.data[i][j] = 0.0;                  // зависимость скорости от β = 0
            }
        }
        time = 0.0;
    }
};

// -----------------------------------------------------------------------------
// Правые части для расширённой системы — уравнения движения и вариаций:
// dr/dt = v
// dv/dt = a(positions)
// d(∂r/∂β)/dt = ∂v/∂β
// d(∂v/∂β)/dt = (∂a/∂r) · (∂r/∂β)
// -----------------------------------------------------------------------------
ExtendedSystemState derivative(const ExtendedSystemState& state, const std::vector<Object>& objects, int asteroid_index) {
    ExtendedSystemState deriv;
    deriv.positions = state.velocities; // dr/dt = v

    auto [accelerations, jacobian] = compute_accelerations_and_jacobian(state.positions, objects, asteroid_index);
    deriv.velocities = accelerations; // dv/dt = a

    // Вариационные уравнения: ∂(∂r/∂β)/dt = ∂v/∂β, ∂(∂v/∂β)/dt = (∂a/∂r)·(∂r/∂β)
    deriv.dr_dbeta = state.dv_dbeta;
    deriv.dv_dbeta = jacobian * state.dr_dbeta;

    return deriv;
}

// -----------------------------------------------------------------------------
// Коэффициенты метода Dormand-Prince (DOPRI5) — используется для фиксированного шага
// Здесь заданы таблицы коэффициентов, затем реализован шаг для расширённой системы
// -----------------------------------------------------------------------------
const std::vector<std::vector<double>> alpha = {
    {1.0/5.0},
    {3.0/40.0, 9.0/40.0},
    {44.0/45.0, -56.0/15.0, 32.0/9.0},
    {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0},
    {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0},
    {35.0/384.0, 0, 500.0/1113.0, 125.0/192.0, 2187.0/6784.0, 11.0/84.0}
};

const std::vector<double> b = {
    35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0
};

const double c_coeffs[6] = {1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0};

// Одиночный шаг DOPRI5 для расширённой системы (фиксированный шаг dt)
void dopri5_step_extended(ExtendedSystemState& state, const std::vector<Object>& objects,
                         int asteroid_index, double dt) {

    ExtendedSystemState k1 = derivative(state, objects, asteroid_index);  // k1

    ExtendedSystemState temp = state + k1 * (alpha[0][0] * dt);
    temp.time = state.time + c_coeffs[0] * dt;
    ExtendedSystemState k2 = derivative(temp, objects, asteroid_index);

    temp = state + (k1 * alpha[1][0] + k2 * alpha[1][1]) * dt;
    temp.time = state.time + c_coeffs[1] * dt;
    ExtendedSystemState k3 = derivative(temp, objects, asteroid_index);

    temp = state + (k1 * alpha[2][0] + k2 * alpha[2][1] + k3 * alpha[2][2]) * dt;
    temp.time = state.time + c_coeffs[2] * dt;
    ExtendedSystemState k4 = derivative(temp, objects, asteroid_index);

    temp = state + (k1 * alpha[3][0] + k2 * alpha[3][1] + k3 * alpha[3][2] + k4 * alpha[3][3]) * dt;
    temp.time = state.time + c_coeffs[3] * dt;
    ExtendedSystemState k5 = derivative(temp, objects, asteroid_index);

    temp = state + (k1 * alpha[4][0] + k2 * alpha[4][1] + k3 * alpha[4][2] + k4 * alpha[4][3] + k5 * alpha[4][4]) * dt;
    temp.time = state.time + c_coeffs[4] * dt;
    ExtendedSystemState k6 = derivative(temp, objects, asteroid_index);

    temp = state + (k1 * alpha[5][0] + k2 * alpha[5][1] + k3 * alpha[5][2] + k4 * alpha[5][3] + k5 * alpha[5][4] + k6 * alpha[5][5]) * dt;
    temp.time = state.time + c_coeffs[5] * dt;
    ExtendedSystemState k7 = derivative(temp, objects, asteroid_index);

    // Обновляем состояние с помощью линейной комбинации ki и коэффициентов b
    ExtendedSystemState new_state;
    new_state.positions.resize(objects.size());
    new_state.velocities.resize(objects.size());

    for (size_t i = 0; i < objects.size(); ++i) {
        new_state.positions[i] = state.positions[i] + (
            k1.positions[i] * b[0] +
            k2.positions[i] * b[1] +
            k3.positions[i] * b[2] +
            k4.positions[i] * b[3] +
            k5.positions[i] * b[4] +
            k6.positions[i] * b[5] +
            k7.positions[i] * b[6]
        ) * dt;

        new_state.velocities[i] = state.velocities[i] + (
            k1.velocities[i] * b[0] +
            k2.velocities[i] * b[1] +
            k3.velocities[i] * b[2] +
            k4.velocities[i] * b[3] +
            k5.velocities[i] * b[4] +
            k6.velocities[i] * b[5] +
            k7.velocities[i] * b[6]
        ) * dt;
    }

    // Обновляем матрицы чувствительности dr_dbeta и dv_dbeta по той же схеме
    new_state.dr_dbeta = state.dr_dbeta + (
        k1.dr_dbeta * b[0] +
        k2.dr_dbeta * b[1] +
        k3.dr_dbeta * b[2] +
        k4.dr_dbeta * b[3] +
        k5.dr_dbeta * b[4] +
        k6.dr_dbeta * b[5] +
        k7.dr_dbeta * b[6]
    ) * dt;

    new_state.dv_dbeta = state.dv_dbeta + (
        k1.dv_dbeta * b[0] +
        k2.dv_dbeta * b[1] +
        k3.dv_dbeta * b[2] +
        k4.dv_dbeta * b[3] +
        k5.dv_dbeta * b[4] +
        k6.dv_dbeta * b[5] +
        k7.dv_dbeta * b[6]
    ) * dt;

    new_state.time = state.time + dt;
    state = new_state; // сохраняем вычисленное новое состояние
}

// -----------------------------------------------------------------------------
// Кубический интерполятор Эрмита для траекторий и для матриц чувствительности
// -----------------------------------------------------------------------------
class CubicHermiteInterpolator {
private:
    std::vector<Vec3d> trajectory; // точки 
    std::vector<double> times;     // соответствующие времена (в секундах)

public:
    CubicHermiteInterpolator(const std::vector<Vec3d>& traj, const std::vector<double>& t)
        : trajectory(traj), times(t) {
        if (trajectory.size() != times.size() || trajectory.size() < 2) {
            throw std::invalid_argument("Неверные данные для интерполятора");
        }
        for (size_t i = 1; i < times.size(); ++i) {
            if (times[i] <= times[i-1]) {
                throw std::invalid_argument("Времена должны строго возрастать");
            }
        }
    }

    // Интерполируем вектор положения в произвольное время.
    // Внешние интервалы обрабатываются как линейные для стабильности.
    Vec3d interpolate(double time) const {
        if (trajectory.empty()) return Vec3d();
        if (time <= times.front()) return trajectory.front();
        if (time >= times.back()) return trajectory.back();

        auto it = std::lower_bound(times.begin(), times.end(), time);
        size_t idx = std::distance(times.begin(), it) - 1;

        // На границах используем простую линейную интерполяцию
        if (idx == 0 || idx >= times.size() - 2) {
            double t0 = times[idx], t1 = times[idx+1];
            Vec3d p0 = trajectory[idx], p1 = trajectory[idx+1];
            double alpha = (time - t0) / (t1 - t0);
            return p0 * (1.0 - alpha) + p1 * alpha;
        }

        return interpolate_hermite(time, idx); // кубическая интерполяция для внутренних точек
    }

    // Статический метод: интерполяция матриц (например dr_dbeta траектории) Эрмитом
    static Mat3x3 interpolate_matrix_hermite(double time, const std::vector<double>& times,
                                            const std::vector<Mat3x3>& matrices) {
        if (matrices.empty()) return Mat3x3();
        if (time <= times.front()) return matrices.front();
        if (time >= times.back()) return matrices.back();

        auto it = std::lower_bound(times.begin(), times.end(), time);
        size_t idx = std::distance(times.begin(), it) - 1;

        // По краям применяем линейную интерполяцию для устойчивости
        if (idx == 0 || idx >= times.size() - 2) {
            double t0 = times[idx], t1 = times[idx+1];
            const Mat3x3& m0 = matrices[idx];
            const Mat3x3& m1 = matrices[idx+1];
            double alpha = (time - t0) / (t1 - t0);
            Mat3x3 result;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result.data[i][j] = m0.data[i][j] * (1.0 - alpha) + m1.data[i][j] * alpha;
            return result;
        }

        // Внутренние точки — кубическая Эрмита с аппроксимацией касательных центральной разностью
        double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1], t3 = times[idx+2];
        const Mat3x3& m0 = matrices[idx-1];
        const Mat3x3& m1 = matrices[idx];
        const Mat3x3& m2 = matrices[idx+1];
        const Mat3x3& m3 = matrices[idx+2];

        Mat3x3 tangent1, tangent2;

        double dt1 = t2 - t0;
        if (std::abs(dt1) < 1e-12) tangent1 = (m2 - m1) * (1.0 / (t2 - t1));
        else tangent1 = (m2 - m0) * (1.0 / dt1);

        double dt2 = t3 - t1;
        if (std::abs(dt2) < 1e-12) tangent2 = (m2 - m1) * (1.0 / (t2 - t1));
        else tangent2 = (m3 - m1) * (1.0 / dt2);

        double u = (time - t1) / (t2 - t1);
        double u2 = u * u;
        double u3 = u2 * u;

        // Эрмитовы базисные функции
        double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        double h10 = u3 - 2.0 * u2 + u;
        double h01 = -2.0 * u3 + 3.0 * u2;
        double h11 = u3 - u2;

        Mat3x3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.data[i][j] = m1.data[i][j] * h00 +
                                   tangent1.data[i][j] * ((t2 - t1) * h10) +
                                   m2.data[i][j] * h01 +
                                   tangent2.data[i][j] * ((t2 - t1) * h11);
            }
        }

        return result;
    }

private:
    // Кубическая Эрмитова интерполяция для векторных траекторий
    Vec3d interpolate_hermite(double time, size_t idx) const {
        double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1], t3 = times[idx+2];
        Vec3d p0 = trajectory[idx-1], p1 = trajectory[idx], p2 = trajectory[idx+1], p3 = trajectory[idx+2];

        double dt1 = t2 - t0;
        double dt2 = t3 - t1;

        Vec3d m1, m2;
        if (std::abs(dt1) < 1e-12) m1 = (p2 - p1) / (t2 - t1);
        else m1 = (p2 - p0) / dt1;

        if (std::abs(dt2) < 1e-12) m2 = (p2 - p1) / (t2 - t1);
        else m2 = (p3 - p1) / dt2;

        double u = (time - t1) / (t2 - t1);
        double u2 = u * u;
        double u3 = u2 * u;

        double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        double h10 = u3 - 2.0 * u2 + u;
        double h01 = -2.0 * u3 + 3.0 * u2;
        double h11 = u3 - u2;

        Vec3d result = p1 * h00 +
                      m1 * ((t2 - t1) * h10) +
                      p2 * h01 +
                      m2 * ((t2 - t1) * h11);

        return result;
    }
};

// -----------------------------------------------------------------------------
// Геометрический якобиан направления (производные RA/DEC по компонентам нормализованного
// вектора направления). Возвращаем матрицу 2x3: строки — [∂RA/∂(x,y,z); ∂DEC/∂(x,y,z)].
// -----------------------------------------------------------------------------
Eigen::Matrix<double, 2, 3> compute_geometric_jacobian_normalized(const Vec3d& direction_unit) {
    Eigen::Matrix<double, 2, 3> J;
    J.setZero();

    double x = direction_unit.x;
    double y = direction_unit.y;
    double z = direction_unit.z;

    double rho2 = x*x + y*y; // квадратичная проекция на XY
    double rho = std::sqrt(rho2);

    // Если проекция на плоскость XY ≈ 0, то RA неопределено; подставляем устойчивые значения
    if (rho < 1e-12) {
        J(0, 0) = 0.0; J(0, 1) = 0.0; J(0, 2) = 0.0;
        J(1, 0) = 0.0; J(1, 1) = 0.0; J(1, 2) = 1.0; // DEC зависит только от z
        return J;
    }

    // ∂RA/∂x, ∂RA/∂y
    J(0, 0) = -y / rho2;
    J(0, 1) =  x / rho2;
    J(0, 2) = 0.0;

    // ∂DEC/∂(x,y,z) 
    double denom = rho2 + z*z; 
    J(1, 0) = -x * z / (rho * denom);
    J(1, 1) = -y * z / (rho * denom);
    J(1, 2) = rho / denom;

    return J;
}

// -----------------------------------------------------------------------------
// Якобиан нормировки: ∂(r/|r|)/∂r = I/|r| - (r r^T)/|r|^3
// Этот якобиан необходим при дифференцировании направления, полученного из топоцентра
// -----------------------------------------------------------------------------
Eigen::Matrix3d compute_normalization_jacobian(const Vec3d& r) {
    Eigen::Matrix3d J_norm;
    J_norm.setZero();

    double r_len = r.length();
    if (r_len < 1e-12) {
        J_norm.setIdentity(); 
        return J_norm;
    }

    double r_len3 = r_len * r_len * r_len;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double ri = (i == 0) ? r.x : (i == 1) ? r.y : r.z;
            double rj = (j == 0) ? r.x : (j == 1) ? r.y : r.z;
            if (i == j) J_norm(i, j) = 1.0 / r_len - ri * rj / r_len3;
            else        J_norm(i, j) = -ri * rj / r_len3;
        }
    }

    return J_norm;
}

// -----------------------------------------------------------------------------
// Якобиан гравитационного отклонения
// -----------------------------------------------------------------------------
Eigen::Matrix3d compute_gravitational_deflection_jacobian_fixed() {
    Eigen::Matrix3d J_grav = Eigen::Matrix3d::Identity();
    return J_grav;
}

// -----------------------------------------------------------------------------
// Более аккуратная оценка якобиана гравитационного отклонения
// -----------------------------------------------------------------------------
Eigen::Matrix3d compute_gravitational_deflection_jacobian_sofa(
    const Vec3d& n,
    const Vec3d& obs_bary,
    const Vec3d& ast_bary,
    const Vec3d& sun_bary) {

    Eigen::Matrix3d J_grav = Eigen::Matrix3d::Identity();

    // Если эффект не включён или мы хотим фиксировать гравитационное отклонение
    if (!APPLY_GRAVITATIONAL_DEFLECTION || FIX_GRAV_DEFLECTION_IN_JACOBIAN) {
        return J_grav;
    }

    // Подготовка векторов для модели 
    double p1[3] = {n.x, n.y, n.z};
    double p2[3] = {0,0,0};

    double e[3] = {obs_bary.x - sun_bary.x, obs_bary.y - sun_bary.y, obs_bary.z - sun_bary.z};
    double em = std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    if (em < 1e-12) return J_grav;

    // Нормируем вектор земля-солнце
    e[0] /= em; e[1] /= em; e[2] /= em;

    // Простая модель: псевдо-угол отклонения от Солнца 
    double gr2 = G * SUN_MASS / (C * C);
    double dt = -gr2 / em;

    double pde = p1[0]*e[0] + p1[1]*e[1] + p1[2]*e[2];
    for (int i = 0; i < 3; ++i)
        p2[i] = p1[i] + dt * (e[i] - pde * p1[i]);

    // Нормировка итогового вектора
    double pm = std::sqrt(p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2]);
    p2[0] /= pm; p2[1] /= pm; p2[2] /= pm;

    // Численный якобиан: центральные разности по компонентам n
    const double eps = 1e-6;
    for (int i = 0; i < 3; ++i) {
        Vec3d n_plus = n;
        Vec3d n_minus = n;

        if (i == 0) { n_plus.x += eps; n_minus.x -= eps; }
        else if (i == 1) { n_plus.y += eps; n_minus.y -= eps; }
        else { n_plus.z += eps; n_minus.z -= eps; }

        n_plus = n_plus.normalized();
        n_minus = n_minus.normalized();

        Vec3d deflected_plus = apply_gravitational_deflection(n_plus, obs_bary, ast_bary, sun_bary);
        Vec3d deflected_minus = apply_gravitational_deflection(n_minus, obs_bary, ast_bary, sun_bary);

        for (int j = 0; j < 3; ++j) {
            double derivative = 0.0;
            if (j == 0) derivative = (deflected_plus.x - deflected_minus.x) / (2.0 * eps);
            else if (j == 1) derivative = (deflected_plus.y - deflected_minus.y) / (2.0 * eps);
            else derivative = (deflected_plus.z - deflected_minus.z) / (2.0 * eps);
            J_grav(j, i) = derivative;
        }
    }

    return J_grav;
}

// -----------------------------------------------------------------------------
// Якобиан релятивистской аберрации 
// -----------------------------------------------------------------------------
Eigen::Matrix3d compute_aberration_jacobian(const Vec3d& n, const Vec3d& observer_velocity_bcrs) {
    Eigen::Matrix3d J_aber = Eigen::Matrix3d::Identity();
    if (!APPLY_ABERRATION) return J_aber;

    // β = v/c — относительные скорости
    Vec3d beta = observer_velocity_bcrs / C;
    double beta2 = beta * beta;
    if (beta2 < 1e-18) return J_aber; // при малых скоростях отличия пренебрежимы

    double gamma = 1.0 / std::sqrt(1.0 - beta2); // релятивистский фактор
    double n_dot_beta = n * beta;

    Eigen::Vector3d n_eig = n.toEigen();
    Eigen::Vector3d beta_eig = beta.toEigen();

    // Аналитические выражения для числителя и знаменателя преобразования
    Eigen::Vector3d term1 = ((gamma - 1.0) / beta2) * n_dot_beta * beta_eig;
    Eigen::Vector3d numerator = n_eig + term1 + gamma * beta_eig;
    double denominator = gamma * (1.0 + n_dot_beta);

    // Производная числителя по n: I + ((γ-1)/β^2) β β^T
    Eigen::Matrix3d dnum_dn = Eigen::Matrix3d::Identity() + ((gamma - 1.0) / beta2) * beta_eig * beta_eig.transpose();

    // Производная знаменателя по n: γ β
    Eigen::Vector3d dden_dn = gamma * beta_eig;

    // Правило дифференцирования дроби: (dnum*den - num*dden^T) / den^2
    J_aber = (dnum_dn * denominator - numerator * dden_dn.transpose()) / (denominator * denominator);

    return J_aber;
}

// -----------------------------------------------------------------------------
// Скорость обсерватории в GCRS, вызванная вращением Земли: v = ω × r
// Здесь r — вектор положения обсерватории в GCRS (топоцентрические координаты)
// -----------------------------------------------------------------------------
Vec3d compute_observatory_velocity_gcrs(const Vec3d& obs_position_gcrs) {
    Vec3d omega(0, 0, EARTH_OMEGA);
    return omega.cross(obs_position_gcrs);
}

// -----------------------------------------------------------------------------
// Гравитационный потенциал от одной точки массы, используется при оценке U_E (скалярный потенциал в выбранной точке)
// -----------------------------------------------------------------------------
double gravitational_potential(const Vec3d& r1, const Vec3d& r2, double mass) {
    Vec3d d = r2 - r1;
    double r = d.length();
    if (r < 1e-10) return 0.0; // защита от деления на ноль
    return G * mass / r;
}

// -----------------------------------------------------------------------------
// Релятивистское преобразование GCRS -> BCRS с учётом поправок 1PN 
// Вход: координаты/скорости в GCRS и орбитальные параметры Земли в BCRS
// Возвращает PV — позицию и скорость в BCRS (аппроксимация IAU 2000/2006 на 1PN)
// -----------------------------------------------------------------------------
PV gcrs_to_bcrs_relativistic(
    const Vec3d& r_gcrs,
    const Vec3d& v_gcrs,
    const Vec3d& rE_bcrs,
    const Vec3d& vE_bcrs,
    double Ue
) {
    const double c2 = C * C;
    const double inv_c2 = 1.0 / c2;

    PV out;

    // Вспомогательные скалярные произведения
    const double vE_dot_rG = vE_bcrs * r_gcrs;
    const double vE_dot_vG = vE_bcrs * v_gcrs;

    // Позиция в BCRS: базовая сумма + корректирующие 1/c^2 члены
    out.r = rE_bcrs
           + r_gcrs
           + vE_bcrs * (0.5 * vE_dot_rG * inv_c2)
           - r_gcrs * (Ue * inv_c2);

    // Скорость в BCRS: аналогично
    out.v = vE_bcrs
           + v_gcrs
           + vE_bcrs * (0.5 * vE_dot_vG * inv_c2)
           - v_gcrs * (Ue * inv_c2);

    return out;
}

// -----------------------------------------------------------------------------
// Модель гравитационного отклонения света Солнцем (линейная аппрокс.)
// Если эффект не включён, возвращаем исходное направление.
// -----------------------------------------------------------------------------
Vec3d apply_gravitational_deflection(const Vec3d& direction_unit, const Vec3d& obs_bary,
                                     const Vec3d& ast_bary, const Vec3d& sun_bary) {
    if (!APPLY_GRAVITATIONAL_DEFLECTION) return direction_unit;

    Vec3d n = direction_unit;
    Vec3d r_sun_obs = sun_bary - obs_bary; // вектор от наблюдателя к Солнцу

    // Проекция перпендикулярно направлению n даёт вектор прицельного параметра
    Vec3d b_vec = r_sun_obs - n * (n * r_sun_obs);
    double b = b_vec.length();
    if (b < 1e-12) return direction_unit; // если наблюдатель и солнце выровнаны — пренебрегаем

    // Простейшая оценка угла отклонения δ = 4GM/(c^2 b)
    double delta = 4.0 * G * SUN_MASS / (C * C * b);
    Vec3d b_hat = b_vec / b;
    Vec3d deflected = n + b_hat * delta; // линейная приближение
    return deflected.normalized();
}

// -----------------------------------------------------------------------------
// Релятивистская аберрация (формулы Лоренца). Возвращаем нормализованный результат.
// При отключении — возвращаем исходный вектор.
// -----------------------------------------------------------------------------
Vec3d apply_aberration(const Vec3d& direction_unit, const Vec3d& observer_velocity_bcrs) {
    if (!APPLY_ABERRATION) return direction_unit;

    Vec3d beta = observer_velocity_bcrs / C;
    double beta2 = beta * beta;
    if (beta2 < 1e-18) return direction_unit; // слишком малая β

    double gamma = 1.0 / std::sqrt(1.0 - beta2);
    double n_dot_beta = direction_unit * beta;

    Vec3d term1 = beta * ((gamma - 1.0) * n_dot_beta / beta2);
    Vec3d numerator = direction_unit + term1 + beta * gamma;
    double denom = gamma * (1.0 + n_dot_beta);

    Vec3d result = numerator / denom;
    return result.normalized();
}

// -----------------------------------------------------------------------------
// Итеративное вычисление поправки за световое время: решаем уравнение
// Δ = |r_ast(t_obs - Δ) - r_obs| / c  методом простых итераций.
// Возвращаем задержку Δ [s]. Если световое время не включено — 0.
// -----------------------------------------------------------------------------
double compute_light_time_correction(double t_obs, const Vec3d& obs_bary,
                                     const CubicHermiteInterpolator& ast_interp,
                                     double tolerance, int max_iter) {
    if (!APPLY_LIGHT_TIME) return 0.0;

    // Простое начальное приближение: расстояние в момент приема / c
    double delta = (ast_interp.interpolate(t_obs) - obs_bary).length() / C;

    for (int iter = 0; iter < max_iter; ++iter) {
        double t_emit = t_obs - delta; // момент излучения
        Vec3d ast_pos = ast_interp.interpolate(t_emit);
        Vec3d r = ast_pos - obs_bary;
        double distance = r.length();

        double new_delta = distance / C;
        if (std::abs(new_delta - delta) < tolerance) return new_delta; // сошлись
        delta = new_delta;
    }

    std::cerr << "Предупреждение: поправка за световое время не сошлась" << std::endl;
    return delta; // возвращаем последнее приближение
}

// -----------------------------------------------------------------------------
// Корректный вид якобиана с учётом зависимости времени испускания от параметров:
// dr_emit/dβ = [I - (v n^T)/(c - n·v)] · (∂r/∂β). При FIX_LIGHT_TIME_IN_JACOBIAN=true
// мы возвращаем незамещённый dr_dbeta_eigen, т.к. фиксируем эффект.
// -----------------------------------------------------------------------------
Eigen::Matrix3d compute_light_time_jacobian(
    double t_obs,
    const Vec3d& obs_bcrs,
    const Vec3d& ast_pos,
    const Vec3d& ast_vel,
    const Eigen::Matrix3d& dr_dbeta_eigen) {

    if (!APPLY_LIGHT_TIME || FIX_LIGHT_TIME_IN_JACOBIAN) {
        return dr_dbeta_eigen; // корректировка выключена — возвращаем как есть
    }

    Eigen::Vector3d r_vec = ast_pos.toEigen() - obs_bcrs.toEigen();
    double r_len = r_vec.norm();
    if (r_len < 1e-12) return dr_dbeta_eigen; // защита

    Eigen::Vector3d n_vec = r_vec / r_len;
    Eigen::Vector3d v_vec = ast_vel.toEigen();

    double n_dot_v = n_vec.dot(v_vec);
    double denom = C - n_dot_v; // важно иметь ненулевой знаменатель
    if (std::abs(denom) < 1e-12) return dr_dbeta_eigen; // случай почти совпадающих направлений

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d correction = I - (v_vec * n_vec.transpose()) / denom;

    return correction * dr_dbeta_eigen;
}

// -----------------------------------------------------------------------------
// Чтение наблюдений из текстового файла
// -----------------------------------------------------------------------------
std::vector<Observation> read_observed_data(const std::string& filename) {
    std::vector<Observation> observations;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл наблюдений: " << filename << std::endl;
        return observations;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // пропускаем пустые строки и комментарии

        std::istringstream ss(line);
        double utc_jd, tdb_jd, ra_deg, dec_deg, ox_km, oy_km, oz_km;
        if (!(ss >> utc_jd >> tdb_jd >> ra_deg >> dec_deg >> ox_km >> oy_km >> oz_km)) {
            std::cerr << "Ошибка чтения строки: " << line << std::endl;
            continue;
        }

        Observation obs;
        obs.time_jd = tdb_jd; 
        obs.time_seconds = jd_to_seconds(tdb_jd);

        obs.ra_observed = ra_deg * DEG_TO_RAD;
        obs.dec_observed = dec_deg * DEG_TO_RAD;

        obs.obs_position_gcrs = Vec3d(ox_km * 1000.0, oy_km * 1000.0, oz_km * 1000.0);

        obs.obs_velocity_gcrs = compute_observatory_velocity_gcrs(obs.obs_position_gcrs);

        observations.push_back(obs);
    }

    file.close();
    return observations;
}

// -----------------------------------------------------------------------------
// Преобразование декартовых координат в небесные (RA/DEC).
// -----------------------------------------------------------------------------
std::pair<double,double> cartesian_to_ra_dec(const Vec3d& v) {
    double ra = std::atan2(v.y, v.x);
    if (ra < 0) ra += 2.0 * PI; // приводим RA к [0,2π)
    double dec = std::atan2(v.z, std::sqrt(v.x * v.x + v.y * v.y));
    return {ra, dec};
}

// -----------------------------------------------------------------------------
// Вычисление прицельного параметра для гравитационного отклонения 
// -----------------------------------------------------------------------------
double compute_impact_parameter(const Vec3d& obs_bary, const Vec3d& ast_bary, const Vec3d& sun_bary) {
    Vec3d r_obs_ast = ast_bary - obs_bary;
    Vec3d r_sun_obs = sun_bary - obs_bary;

    Vec3d crossp = r_obs_ast.cross(r_sun_obs);
    double area = crossp.length();
    double base = r_obs_ast.length();
    if (base < 1e-12) return 1e20; // защита
    return area / base;
}

// -----------------------------------------------------------------------------
// Создаём модель системы из нескольких тел: астероид, Солнце, Юпитер, Земля. Возвращаем вектор объектов.
// -----------------------------------------------------------------------------
std::vector<Object> create_model_system(const Vec3d& asteroid_position, const Vec3d& asteroid_velocity) {
    std::vector<Object> objects;

    const double EARTH_MASS = 5.972e24;
    const double JUPITER_MASS = 1.898e27;

    // Астероид — объект
    Object kobresia("Kobresia", 3.0e16, 0, true);
    kobresia.position = asteroid_position;
    kobresia.velocity = asteroid_velocity;

    // Солнце 
    Object sun("Sun", SUN_MASS, 1, false);
    sun.position = Vec3d(-1.359633615694579E+06,  1.163993750658928E+05,  8.372652597099889E+04) * 1000.0;
    sun.velocity = Vec3d(-3.380122006909895E-04, -1.447967375853807E-02, -6.132577922009313E-03) * 1000.0;

    // Юпитер
    Object jupiter("Jupiter", JUPITER_MASS, 2, false);
    jupiter.position = Vec3d(7.381426304618969E+08,  5.029757079706200E+07,  3.592074173013674E+06) * 1000.0;
    jupiter.velocity = Vec3d(-9.927319384212236E-01,  1.253707948797421E+01,  5.397847549901059E+00) * 1000.0;

    // Земля
    Object earth("Earth", EARTH_MASS, 3, false);
    earth.position = Vec3d(1.475204020619656E+08,  1.570888911282263E+07,  6.842172695828106E+06) * 1000.0;
    earth.velocity = Vec3d(-3.872927841529505E+00,  2.704280911152226E+01,  1.172423083084365E+01) * 1000.0;

    objects.push_back(kobresia);
    objects.push_back(sun);
    objects.push_back(jupiter);
    objects.push_back(earth);

    return objects;
}

// -----------------------------------------------------------------------------
// Инициализация расширенного состояния перед интегрированием
// -----------------------------------------------------------------------------
void init_extended_state(const std::vector<Object>& objects, ExtendedSystemState& state, int asteroid_index) {
    state.positions.clear();
    state.velocities.clear();

    for (const auto& obj : objects) {
        state.positions.push_back(obj.position);
        state.velocities.push_back(obj.velocity);
    }

    // dr_dbeta = I, dv_dbeta = 0 по умолчанию — уже задано в clear() при необходимости
}

// -----------------------------------------------------------------------------
// Интегрирование расширённой системы на равномерной сетке времени [t_start, t_end] с шагом dt. Результат — вектор состояний states (включая начальное состояние).
// -----------------------------------------------------------------------------
void integrate_extended_system(const std::vector<Object>& objects, double t_start, double t_end,
                              double dt, std::vector<ExtendedSystemState>& states, int asteroid_index) {
    ExtendedSystemState current_state;
    init_extended_state(objects, current_state, asteroid_index);
    current_state.time = t_start;

    states.clear();
    states.push_back(current_state);

    int total_steps = static_cast<int>((t_end - t_start) / dt);
    for (int step = 1; step <= total_steps; ++step) {
        dopri5_step_extended(current_state, objects, asteroid_index, dt);
        states.push_back(current_state);
    }
}

// -----------------------------------------------------------------------------
// РЕШЕНИЕ ОБРАТНОЙ ЗАДАЧИ: задача на параметризацию начальной позиции астероида β
// Используем метод Гаусса-Ньютона: формируем матрицу Якоби A и вектор невязок r,
// решаем нормальные уравнения (A^T A) Δβ = A^T r
// -----------------------------------------------------------------------------
Vec3d solve_inverse_problem(const std::vector<Observation>& observations,
                           const Vec3d& initial_guess,
                           const Vec3d& fixed_velocity,
                           double t_start, double t_end, double dt) {

    std::cout << "\n=========================================\n";
    std::cout << "РЕШЕНИЕ ОБРАТНОЙ ЗАДАЧИ МЕТОДОМ ГАУССА-НЬЮТОНА\n";
    std::cout << "=========================================\n\n";

    // beta — оценка параметров
    Vec3d beta = initial_guess;

    std::ofstream log_file("residuals_ra_dec.csv");
    log_file << std::setprecision(12);
    log_file << "# iteration, obs_index, jd_tdb, dra_rad, ddec_rad\n";

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::cout << "\n--- Итерация " << iteration + 1 << " из " << MAX_ITERATIONS << " ---" << std::endl;
        std::cout << "Текущая оценка beta: (" << beta.x << ", " << beta.y << ", " << beta.z << ") m" << std::endl;

        // 1) Прямое интегрирование системы для текущей оценки beta
        std::vector<Object> system = create_model_system(beta, fixed_velocity);
        std::vector<ExtendedSystemState> states;
        integrate_extended_system(system, t_start, t_end, dt, states, 0);

        // 2) Подготовим траектории/матрицы чувствительности и времена для интерполяции
        std::vector<Vec3d> asteroid_traj, earth_traj, sun_traj, jupiter_traj;
        std::vector<Vec3d> asteroid_vel_traj, earth_vel_traj;
        std::vector<Mat3x3> dr_dbeta_traj;
        std::vector<double> times;

        for (const auto& state : states) {
            asteroid_traj.push_back(state.positions[0]);
            asteroid_vel_traj.push_back(state.velocities[0]);
            earth_traj.push_back(state.positions[3]);
            sun_traj.push_back(state.positions[1]);
            jupiter_traj.push_back(state.positions[2]);
            earth_vel_traj.push_back(state.velocities[3]);
            dr_dbeta_traj.push_back(state.dr_dbeta);
            times.push_back(state.time);
        }

        // Создаём интерполяторы для получения значений в моменты наблюдений
        CubicHermiteInterpolator asteroid_interp(asteroid_traj, times);
        CubicHermiteInterpolator asteroid_vel_interp(asteroid_vel_traj, times);
        CubicHermiteInterpolator earth_interp(earth_traj, times);
        CubicHermiteInterpolator sun_interp(sun_traj, times);
        CubicHermiteInterpolator jupiter_interp(jupiter_traj, times);
        CubicHermiteInterpolator earth_vel_interp(earth_vel_traj, times);

        // 3) Матрица Якоби A 
        int num_obs = observations.size();
        Eigen::MatrixXd A(2 * num_obs, 3);
        Eigen::VectorXd r(2 * num_obs);
        A.setZero(); r.setZero();

        // 4) Проходим по наблюдениям, формируем r и строки матрицы A
        for (int i = 0; i < num_obs; ++i) {
            const Observation& obs = observations[i];

            // Получаем состояние Земли/Солнца/Юпитера в момент приёма
            Vec3d earth_pos = earth_interp.interpolate(obs.time_seconds);
            Vec3d earth_vel = earth_vel_interp.interpolate(obs.time_seconds);
            Vec3d sun_pos = sun_interp.interpolate(obs.time_seconds);
            Vec3d jupiter_pos = jupiter_interp.interpolate(obs.time_seconds);

            // Оценка гравитационного потенциала в точке наблюдателя (упрощённо суммируем вклады)
            double Ue = gravitational_potential(earth_pos, sun_pos, SUN_MASS)
                      + gravitational_potential(earth_pos, jupiter_pos, JUPITER_MASS);

            // Преобразуем позицию/скорость обсерватории GCRS -> BCRS 
            PV obs_bary = gcrs_to_bcrs_relativistic(
                obs.obs_position_gcrs,
                obs.obs_velocity_gcrs,
                earth_pos,
                earth_vel,
                Ue
            );

            Vec3d obs_bcrs = obs_bary.r;
            Vec3d obs_vel_bcrs = obs_bary.v;

            // Поправка за световое время 
            double light_time = compute_light_time_correction(obs.time_seconds, obs_bcrs, asteroid_interp);
            double t_emit = obs.time_seconds - light_time;

            // Положение и скорость астероида в момент испускания 
            Vec3d ast_pos = asteroid_interp.interpolate(t_emit);
            Vec3d ast_vel = asteroid_vel_interp.interpolate(t_emit);

            // Топоцентрический вектор от наблюдателя к астероиду 
            Vec3d topocentric = ast_pos - obs_bcrs;
            Vec3d n = topocentric.normalized();

            // Если включено, применяем гравитационное отклонение (Солнце) и аберрацию
            Vec3d n_grav = n;
            if (APPLY_GRAVITATIONAL_DEFLECTION) n_grav = apply_gravitational_deflection(n, obs_bcrs, ast_pos, sun_pos);

            Vec3d n_aber = n_grav;
            if (APPLY_ABERRATION) n_aber = apply_aberration(n_grav, obs_vel_bcrs);

            Vec3d direction_corrected = n_aber; // окончательное направление

            // Вычисляем RA/DEC
            auto comp_ang = cartesian_to_ra_dec(direction_corrected);
            double ra_computed = comp_ang.first;
            double dec_computed = comp_ang.second;

            // Невязки 
            double dra = obs.ra_observed - ra_computed;
            double ddec = obs.dec_observed - dec_computed;

            // Приводим dra к диапазону [-π, π]
            while (dra > PI) dra -= 2.0 * PI;
            while (dra < -PI) dra += 2.0 * PI;

            // Логируем невязки для последующего анализа
            log_file << iteration << "," << i << "," << obs.time_jd << "," << dra << "," << ddec << "\n";

            r(2*i)   = dra;
            r(2*i+1) = ddec;

            // Формируем строку матрицы A для данного наблюдения
            // 1) Геометрическая часть: производные RA/DEC по компонентам нормализованного вектора
            Eigen::Matrix<double, 2, 3> J_geom = compute_geometric_jacobian_normalized(direction_corrected);

            // 2) Якобиан нормировки топоцентра (∂(r/|r|)/∂r)
            Eigen::Matrix3d J_norm = compute_normalization_jacobian(topocentric);

            Eigen::Matrix3d J_dir = Eigen::Matrix3d::Identity();
            J_dir = J_norm * J_dir; // применяем только нормировку
            
            Eigen::Matrix<double, 2, 3> J_geom_full = J_geom * J_dir;

            // Матрица чувствительности ∂r/∂β в момент испускания 
            Mat3x3 dr_dbeta_emit = CubicHermiteInterpolator::interpolate_matrix_hermite(t_emit, times, dr_dbeta_traj);
            Eigen::Matrix3d dr_dbeta_eigen = dr_dbeta_emit.toEigen();

            Eigen::Matrix3d dr_dbeta_light = compute_light_time_jacobian(obs.time_seconds, obs_bcrs, ast_pos, ast_vel, dr_dbeta_eigen);

            // Итоговая строка матрицы A: ∂(angles)/∂β = - J_geom_full * dr_dbeta_light
            Eigen::Matrix<double, 2, 3> A_i = -J_geom_full * dr_dbeta_light;

            // Копируем в глобальную матрицу A
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 3; ++col)
                    A(2*i + row, col) = A_i(row, col);
        }
        Eigen::Matrix3d ATA = A.transpose() * A;
        Eigen::Vector3d ATr = A.transpose() * r;

        // Оценка числа обусловленности через SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
        double cond = svd.singularValues()(0) / svd.singularValues()(2);

        std::cout << "Число обусловленности ATA: " << cond << std::endl;

        Eigen::Vector3d delta_beta_eigen;
        if (cond > MAX_CONDITION_NUMBER) {
            std::cout << "Предупреждение: плохая обусловленность матрицы!" << std::endl;
            double lambda_adjusted = TIKHONOV_LAMBDA * std::sqrt(cond);
            std::cout << "Увеличиваем регуляризацию до: " << lambda_adjusted << std::endl;

            Eigen::Matrix3d reg = lambda_adjusted * Eigen::Matrix3d::Identity();
            Eigen::Matrix3d ATA_reg = ATA + reg;

            // Решаем регуляризованную систему
            delta_beta_eigen = ATA_reg.ldlt().solve(ATr);
        } else {
            Eigen::Matrix3d reg = TIKHONOV_LAMBDA * Eigen::Matrix3d::Identity();
            Eigen::Matrix3d ATA_reg = ATA + reg;
            delta_beta_eigen = ATA_reg.ldlt().solve(ATr);
        }

        Vec3d delta_beta(delta_beta_eigen(0), delta_beta_eigen(1), delta_beta_eigen(2));

        // Контроль размера шага: если норма слишком велика — уменьшаем демпфированием
        double step_norm = delta_beta.length();
        double damping = DAMPING_FACTOR;
        if (step_norm > MAX_STEP_NORM) {
            damping = DAMPING_FACTOR * (MAX_STEP_NORM / step_norm);
            std::cout << "Предупреждение: норма шага слишком большая (" << step_norm << " m), уменьшаем демпфирование до " << damping << std::endl;
        }

        Vec3d beta_new = beta - delta_beta * damping; // поправка параметра (знак минус — по формуле метода)

        std::cout << "Поправка Δβ: (" << delta_beta.x << ", " << delta_beta.y << ", " << delta_beta.z << ") m" << std::endl;
        std::cout << "Норма Δβ: " << step_norm << " m (" << step_norm / AU << " AU)" << std::endl;
        std::cout << "Коэффициент демпфирования: " << damping << std::endl;

        // Проверка сходимости по норме поправки
        if (step_norm < CONVERGENCE_TOLERANCE) {
            std::cout << "\n✓ СХОДИМОСТЬ ДОСТИГНУТА НА ИТЕРАЦИИ " << iteration + 1 << std::endl;
            beta = beta_new;
            break;
        }

        beta = beta_new; // обновляем оценку

        // Вывод RMS статистики по невязкам (в угловых секундах)
        double rms_ra = 0.0, rms_dec = 0.0;
        for (int i = 0; i < num_obs; ++i) {
            rms_ra += r(2*i) * r(2*i);
            rms_dec += r(2*i + 1) * r(2*i + 1);
        }
        rms_ra = std::sqrt(rms_ra / num_obs) * RAD_TO_ARCSEC;
        rms_dec = std::sqrt(rms_dec / num_obs) * RAD_TO_ARCSEC;

        std::cout << "RMS невязки: RA = " << rms_ra << " угл.сек, DEC = " << rms_dec << " угл.сек" << std::endl;
    }

    log_file.close();
    return beta;
}

int main() {
    // Входные параметры интегрирования: JD начала, длительность (дни) и шаг в секундах
    double start_jd = 2459852.5;  // примерное начальное время JD
    double duration_days = 9.0;   // интервал интегрирования [дней]
    double dt = 60.0;             // шаг интегрирования [s]

    double t_start = jd_to_seconds(start_jd);
    double t_end = t_start + duration_days * 86400.0;

    // Считываем таблицу наблюдений (output.txt) — формат описан в read_observed_data
    std::vector<Observation> observations = read_observed_data("output.txt");
    if (observations.empty()) {
        std::cerr << "Нет наблюдений для обработки." << std::endl;
        return 1;
    }

    std::cout << "Загружено " << observations.size() << " наблюдений." << std::endl;

    // Начальное приближение (позиция) и фиксированная скорость астероида — заданы в метрах/м/с
    Vec3d initial_guess = Vec3d(-1.247231001936561E+08, 3.501040899062184E+08, 1.544614456531207E+08) * 1000.0;
    Vec3d fixed_velocity = Vec3d(-1.663615216546555E+01, -4.632952969649452E+00, -3.889776207120005E+00) * 1000.0;

    std::cout << "\nНачальное приближение:" << std::endl;
    initial_guess.print("  Положение");
    fixed_velocity.print("  Скорость (фиксированная)");

    // Запускаем решение обратной задачи — получаем оценку начальной позиции астероида
    Vec3d estimated_position = solve_inverse_problem(
        observations, initial_guess, fixed_velocity, t_start, t_end, dt);

    // Выводим результаты и сравнение с исходным приближением
    std::cout << "\n=========================================\n";
    std::cout << "ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ\n";
    std::cout << "=========================================\n\n";

    std::cout << "Начальное положение астероида:" << std::endl;
    estimated_position.print("  Оцененное");
    initial_guess.print("  Исходное ");

    Vec3d diff = estimated_position - initial_guess;
    std::cout << "Разница: " << diff.length() << " м" << std::endl;
    std::cout << "В астрономических единицах: " << diff.length() / AU << " AU\n" << std::endl;

    return 0;
}
