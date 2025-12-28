#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <sofa.h>

// Константы
const double G = 6.67430e-11;
const double PI = 3.14159265358979323846;
const double C = 299792458.0; 
const double SUN_MASS = 1.989e30; 
const double AU = 149597870700.0; 
const double DEG_TO_RAD = PI / 180.0;
const double RAD_TO_DEG = 180.0 / PI;
const double ARCSEC_TO_RAD = PI / (180.0 * 3600.0);
const double RAD_TO_ARCSEC = 180.0 * 3600.0 / PI;

// Параметры для поправок
const bool APPLY_GRAVITATIONAL_DEFLECTION = true;
const bool APPLY_ABERRATION = true;
const bool APPLY_LIGHT_TIME = true;

// Земные параметры
const double EARTH_RADIUS = 6378137.0; 
const double EARTH_OMEGA = 7.2921150e-5; 

// Структуры данных
struct Vec3d {
    double x, y, z;
    
    Vec3d() : x(0), y(0), z(0) {}
    Vec3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vec3d operator+(const Vec3d& other) const {
        return Vec3d(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3d operator-(const Vec3d& other) const {
        return Vec3d(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3d operator*(double scalar) const {
        return Vec3d(x * scalar, y * scalar, z * scalar);
    }
    
    Vec3d operator/(double scalar) const {
        return Vec3d(x / scalar, y / scalar, z / scalar);
    }
    
    double operator*(const Vec3d& other) const { 
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vec3d cross(const Vec3d& other) const {
        return Vec3d(y * other.z - z * other.y,
                    z * other.x - x * other.z,
                    x * other.y - y * other.x);
    }
    
    double length() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    double length_squared() const {
        return x*x + y*y + z*z;
    }
    
    Vec3d normalized() const {
        double len = length();
        if (len < 1e-15) return *this;
        return *this / len;
    }
    
    void print(const std::string& name) const {
        std::cout << std::setprecision(12) << name << ": (" << x << ", " << y << ", " << z << ")" << std::endl;
    }
};

// Структура для позиции и скорости
struct PV {
    Vec3d r; // position [m]
    Vec3d v; // velocity [m/s]
};

struct Object {
    Vec3d position;
    Vec3d velocity;
    double mass;
    std::string name;
    
    Object() : mass(0), name("") {}
    Object(const std::string& n, double m) : mass(m), name(n) {}
};

struct SystemState {
    std::vector<Vec3d> positions;
    std::vector<Vec3d> velocities;
    double time = 0.0;
    
    SystemState operator+(const SystemState& other) const {
        SystemState result;
        result.positions.resize(positions.size());
        result.velocities.resize(velocities.size());
        
        for (size_t i = 0; i < positions.size(); ++i) {
            result.positions[i] = positions[i] + other.positions[i];
            result.velocities[i] = velocities[i] + other.velocities[i];
        }
        return result;
    }
    
    SystemState operator*(double scalar) const {
        SystemState result;
        result.positions.resize(positions.size());
        result.velocities.resize(velocities.size());
        
        for (size_t i = 0; i < positions.size(); ++i) {
            result.positions[i] = positions[i] * scalar;
            result.velocities[i] = velocities[i] * scalar;
        }
        return result;
    }
    
    void clear() {
        positions.clear();
        velocities.clear();
        time = 0.0;
    }
};

// Структура для хранения данных наблюдений
struct Observation {
    double time_jd;               // Время в JD (TDB)
    double time_seconds;          // Время в секундах от J2000
    double ra_observed;           // Наблюдаемое RA в радианах
    double dec_observed;          // Наблюдаемое DEC в радианах
    Vec3d obs_position_gcrs;      // Положение обсерватории в GCRS (м)
    Vec3d obs_position_bcrs;      // Положение обсерватории в BCRS (рассчитанное)
    Vec3d obs_velocity_gcrs;      // Скорость обсерватории в GCRS (м/с)
    Vec3d obs_velocity_bcrs;      // Скорость обсерватории в BCRS (рассчитанное)
    
    void print() const {
        std::cout << "\nНаблюдение (JD " << time_jd << "):" << std::endl;
        std::cout << "  RA: " << ra_observed * RAD_TO_DEG << "°, DEC: " << dec_observed * RAD_TO_DEG << "°" << std::endl;
        std::cout << "  Обсерватория GCRS (м): (" << obs_position_gcrs.x << ", " 
                  << obs_position_gcrs.y << ", " << obs_position_gcrs.z << ")" << std::endl;
        std::cout << "  Скорость обсерватории GCRS (м/с): (" << obs_velocity_gcrs.x << ", "
                  << obs_velocity_gcrs.y << ", " << obs_velocity_gcrs.z << ")" << std::endl;
    }
};

// Коэффициенты метода Дорманда-Принса 5-го порядка
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

// Коэффициенты c для метода Дорманда-Принса
const double c_coeffs[6] = {1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0};

// Преобразования времени
double jd_to_seconds(double jd) {
    const double j2000_jd = 2451545.0;
    return (jd - j2000_jd) * 86400.0;
}

double seconds_to_jd(double seconds) {
    const double j2000_jd = 2451545.0;
    return j2000_jd + seconds / 86400.0;
}

// Функция вычисления ускорений из состояния
std::vector<Vec3d> compute_accelerations_from_state(const SystemState& state, const std::vector<Object>& objects) {
    size_t n = objects.size();
    std::vector<Vec3d> accelerations(n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            
            Vec3d r = state.positions[j] - state.positions[i];
            double dist2 = r * r;
            double dist = std::sqrt(dist2);
            
            if (dist < 1e-10) continue;
            
            Vec3d dir = r / dist;
            accelerations[i] = accelerations[i] + (dir * (G * objects[j].mass / dist2));
        }
    }
    
    return accelerations;
}

// Производные
SystemState derivative(const SystemState& state, const std::vector<Object>& objects) {
    SystemState deriv;
    deriv.positions = state.velocities;
    deriv.velocities = compute_accelerations_from_state(state, objects);
    return deriv;
}

// Метод Дорманда-Принса
void dopri5_step(SystemState& state, const std::vector<Object>& objects, double dt) {
    SystemState k1 = derivative(state, objects);
    
    SystemState temp = state + k1 * (alpha[0][0] * dt);
    temp.time = state.time + c_coeffs[0] * dt;
    SystemState k2 = derivative(temp, objects);
    
    temp = state + (k1 * alpha[1][0] + k2 * alpha[1][1]) * dt;
    temp.time = state.time + c_coeffs[1] * dt;
    SystemState k3 = derivative(temp, objects);
    
    temp = state + (k1 * alpha[2][0] + k2 * alpha[2][1] + k3 * alpha[2][2]) * dt;
    temp.time = state.time + c_coeffs[2] * dt;
    SystemState k4 = derivative(temp, objects);
    
    temp = state + (k1 * alpha[3][0] + k2 * alpha[3][1] + k3 * alpha[3][2] + k4 * alpha[3][3]) * dt;
    temp.time = state.time + c_coeffs[3] * dt;
    SystemState k5 = derivative(temp, objects);
    
    temp = state + (k1 * alpha[4][0] + k2 * alpha[4][1] + k3 * alpha[4][2] + k4 * alpha[4][3] + k5 * alpha[4][4]) * dt;
    temp.time = state.time + c_coeffs[4] * dt;
    SystemState k6 = derivative(temp, objects);
    
    temp = state + (k1 * alpha[5][0] + k2 * alpha[5][1] + k3 * alpha[5][2] + k4 * alpha[5][3] + k5 * alpha[5][4] + k6 * alpha[5][5]) * dt;
    temp.time = state.time + c_coeffs[5] * dt;
    SystemState k7 = derivative(temp, objects);
    
    SystemState new_state;
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
    
    new_state.time = state.time + dt;
    state = new_state;
}

// Инициализация состояния
void init_state(const std::vector<Object>& objects, SystemState& state) {
    state.positions.clear();
    state.velocities.clear();
    
    for (const auto& obj : objects) {
        state.positions.push_back(obj.position);
        state.velocities.push_back(obj.velocity);
    }
}

// Класс для кубической интерполяции Эрмита (Catmull-Rom)
class CubicHermiteInterpolator {
public:
    CubicHermiteInterpolator(const std::vector<Vec3d>& trajectory, const std::vector<double>& times) 
        : trajectory(trajectory), times(times) {
        // Проверяем корректность входных данных
        if (trajectory.size() != times.size() || trajectory.size() < 2) {
            throw std::invalid_argument("Неверные данные для интерполятора: размеры не совпадают или недостаточно точек");
        }
        
        // Проверяем строгое возрастание времени
        for (size_t i = 1; i < times.size(); ++i) {
            if (times[i] <= times[i-1]) {
                throw std::invalid_argument("Времена должны строго возрастать");
            }
        }
    }
    
    Vec3d interpolate(double time) const {
        if (trajectory.empty()) return Vec3d();
        if (time <= times.front()) return trajectory.front();
        if (time >= times.back()) return trajectory.back();
        
        // Находим индекс интервала
        auto it = std::lower_bound(times.begin(), times.end(), time);
        size_t idx = std::distance(times.begin(), it) - 1;
        
        // Для интервалов у краев используем линейную интерполяцию
        if (idx == 0 || idx >= times.size() - 2) {
            double t0 = times[idx], t1 = times[idx+1];
            Vec3d p0 = trajectory[idx], p1 = trajectory[idx+1];
            
            double alpha = (time - t0) / (t1 - t0);
            return p0 * (1.0 - alpha) + p1 * alpha;
        }
        
        // Кубическая интерполяция Эрмита для внутренних точек
        return interpolate_hermite(time, idx);
    }
    
private:
    Vec3d interpolate_hermite(double time, size_t idx) const {
        // Точки: p0, p1, p2, p3
        double t0 = times[idx-1], t1 = times[idx], t2 = times[idx+1], t3 = times[idx+2];
        Vec3d p0 = trajectory[idx-1], p1 = trajectory[idx], p2 = trajectory[idx+1], p3 = trajectory[idx+2];
        
        // Вычисляем касательные 
        double dt1 = t2 - t0;
        double dt2 = t3 - t1;
        
        Vec3d m1, m2;
        if (std::abs(dt1) < 1e-12) {
            // Если интервал слишком мал, используем одностороннюю разность
            m1 = (p2 - p1) / (t2 - t1);
        } else {
            m1 = (p2 - p0) / dt1;
        }
        
        if (std::abs(dt2) < 1e-12) {
            // Если интервал слишком мал, используем одностороннюю разность
            m2 = (p2 - p1) / (t2 - t1);
        } else {
            m2 = (p3 - p1) / dt2;
        }
        
        // Нормализованный параметр u ∈ [0, 1]
        double u = (time - t1) / (t2 - t1);
        double u2 = u * u;
        double u3 = u2 * u;
        
        // Базисные функции Эрмита
        double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        double h10 = u3 - 2.0 * u2 + u;
        double h01 = -2.0 * u3 + 3.0 * u2;
        double h11 = u3 - u2;
        
        // Интерполяция
        Vec3d result = p1 * h00 + 
                      m1 * ((t2 - t1) * h10) + 
                      p2 * h01 + 
                      m2 * ((t2 - t1) * h11);
        
        return result;
    }
    
    std::vector<Vec3d> trajectory;
    std::vector<double> times;
};

// Вычисление скорости обсерватории в GCRS из-за вращения Земли
Vec3d compute_observatory_velocity_gcrs(const Vec3d& obs_position_gcrs) {
    // Угловая скорость Земли (вектор направлен вдоль оси вращения)
    Vec3d omega(0, 0, EARTH_OMEGA);
    
    // Скорость точки на поверхности: v = ω × r
    return omega.cross(obs_position_gcrs);
}

//Вычисление ньютоновского гравитационного потенциала Солнца в точке Земли
double earth_gravitational_potential(const Vec3d& rE_bcrs, const Vec3d& rSun_bcrs) {
    Vec3d d = rSun_bcrs - rE_bcrs;
    double r = d.length();
    if (r < 1e-10) return 0.0; 
    return G * SUN_MASS / r; // m^2/s^2
}

// High-accuracy GCRS → BCRS transformation (IAU 2000/2006, 1PN)
PV gcrs_to_bcrs_relativistic(
    const Vec3d& r_gcrs,
    const Vec3d& v_gcrs,
    const Vec3d& rE_bcrs,
    const Vec3d& vE_bcrs,
    double Ue
) {
    const double c2 = C * C;
    const double inv_c2 = 1.0 / c2;
    
    // Скалярные произведения
    const double vE_dot_rG = vE_bcrs * r_gcrs;
    const double vE_dot_vG = vE_bcrs * v_gcrs;
    
    PV out;
    
    // Позиция: r_BCRS = r_E + r_G + (1/c²)[ (1/2)v_E(v_E·r_G) - U_E r_G ]
    out.r = rE_bcrs 
           + r_gcrs 
           + vE_bcrs * (0.5 * vE_dot_rG * inv_c2)
           - r_gcrs * (Ue * inv_c2);
    
    // Скорость: v_BCRS = v_E + v_G + (1/c²)[ (1/2)v_E(v_E·v_G) - U_E v_G ]
    out.v = vE_bcrs
           + v_gcrs
           + vE_bcrs * (0.5 * vE_dot_vG * inv_c2)
           - v_gcrs * (Ue * inv_c2);
    
    return out;
}

// Чтение наблюдений с новым форматом
std::vector<Observation> read_observed_data(const std::string& filename) {
    std::vector<Observation> observations;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл наблюдений: " << filename << std::endl;
        return observations;
    }
    
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream ss(line);
        double utc, tdb, ra_deg, dec_deg, ox_km, oy_km, oz_km;
        
        if (!(ss >> utc >> tdb >> ra_deg >> dec_deg >> ox_km >> oy_km >> oz_km)) {
            std::cerr << "Ошибка чтения строки: " << line << std::endl;
            continue;
        }
        
        Observation obs;
        obs.time_jd = tdb;  
        obs.time_seconds = jd_to_seconds(tdb);
        
        obs.ra_observed = ra_deg * DEG_TO_RAD;
        obs.dec_observed = dec_deg * DEG_TO_RAD;
        
        // Координаты обсерватории: км → м 
        obs.obs_position_gcrs = Vec3d(ox_km * 1000.0, oy_km * 1000.0, oz_km * 1000.0);
        
        // Скорость обсерватории в GCRS
        obs.obs_velocity_gcrs = compute_observatory_velocity_gcrs(obs.obs_position_gcrs);
        
        observations.push_back(obs);
        count++;
    }
    
    file.close();
    std::cout << "Прочитано наблюдений: " << count << std::endl;
    return observations;
}

// Конвертация в небесные координаты RA/DEC
std::pair<double,double> cartesian_to_ra_dec(const Vec3d& v) {
    double ra = std::atan2(v.y, v.x);
    if (ra < 0) ra += 2.0 * PI;
    double dec = std::atan2(v.z, std::sqrt(v.x * v.x + v.y * v.y));
    return {ra, dec};
}

// Вычисление прицельного параметра для гравитационного отклонения
double compute_impact_parameter(const Vec3d& obs_bary, const Vec3d& ast_bary, const Vec3d& sun_bary) {
    Vec3d r_obs_ast = ast_bary - obs_bary; // вектор от наблюдателя к астероиду
    Vec3d r_sun_obs = sun_bary - obs_bary; // вектор от наблюдателя к Солнцу
    
    Vec3d crossp = r_obs_ast.cross(r_sun_obs);
    double area = crossp.length();
    
    double base = r_obs_ast.length();
    
    if (base < 1e-12) return 1e20;
    
    // Расстояние от точки (Солнца) до прямой (наблюдатель-астероид)
    return area / base;
}

// Функция для вычисления гравитационного отклонения света Солнцем
Vec3d apply_gravitational_deflection(const Vec3d& direction_unit, const Vec3d& obs_bary, 
                                     const Vec3d& ast_bary, const Vec3d& sun_bary) {
    if (!APPLY_GRAVITATIONAL_DEFLECTION) return direction_unit;
    
    // Вычисляем прицельный параметр
    double impact_parameter = compute_impact_parameter(obs_bary, ast_bary, sun_bary);
    
    if (impact_parameter > 1e19 || impact_parameter < 1e-12) return direction_unit;
    
    double deflection_angle = 4.0 * G * SUN_MASS / (C * C * impact_parameter);
    
    if (deflection_angle < 1e-15) return direction_unit;
    
    // Находим ось вращения: перпендикуляр к плоскости, содержащей луч и Солнце
    Vec3d sun_to_obs = sun_bary - obs_bary;
    Vec3d axis = direction_unit.cross(sun_to_obs);
    double axis_len = axis.length();
    
    if (axis_len < 1e-12) {
        // Луч идет почти прямо на Солнце или от него
        return direction_unit;
    }
    
    axis = axis / axis_len;
    
    // Вращаем вектор направления на угол отклонения
    double cos_theta = std::cos(deflection_angle);
    double sin_theta = std::sin(deflection_angle);
    
    Vec3d result = direction_unit * cos_theta + 
                   axis.cross(direction_unit) * sin_theta + 
                   axis * (axis * direction_unit) * (1.0 - cos_theta);
    
    return result.normalized();
}

// Функция для вычисления релятивистской аберрации света
Vec3d apply_aberration(const Vec3d& direction_unit, const Vec3d& observer_velocity_bcrs) {
    if (!APPLY_ABERRATION) return direction_unit;
    
    Vec3d beta = observer_velocity_bcrs / C; // вектор β = v/c
    double beta2 = beta * beta;
    
    if (beta2 < 1e-18) return direction_unit;
    
    double gamma = 1.0 / std::sqrt(1.0 - beta2);
    double n_dot_beta = direction_unit * beta;
    
    // Релятивистская формула аберрации 
    // n' = (n + (γ-1)(n·β)β/β² + γβ) / (γ(1 + n·β))
    Vec3d term1 = beta * ((gamma - 1.0) * n_dot_beta / beta2);
    Vec3d numerator = direction_unit + term1 + beta * gamma;
    double denom = gamma * (1.0 + n_dot_beta);
    
    Vec3d result = numerator / denom;
    return result.normalized();
}

// Функция для вычисления поправки за световое время (итерационная)
double compute_light_time_correction(double t_obs, const Vec3d& obs_bary,
                                     const CubicHermiteInterpolator& ast_interp,
                                     double tolerance = 1e-9, int max_iter = 20) {
    if (!APPLY_LIGHT_TIME) return 0.0;
    
    double delta = 0.0;
    
    // Начальное приближение: предполагаем, что астероид в момент испускания находится в той же позиции, что и в момент приема
    delta = (ast_interp.interpolate(t_obs) - obs_bary).length() / C;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double t_emit = t_obs - delta;
        
        // Положение астероида в момент испускания
        Vec3d ast_pos = ast_interp.interpolate(t_emit);
        
        // Расстояние между астероидом и наблюдателем
        Vec3d r = ast_pos - obs_bary;
        double distance = r.length();
        
        double new_delta = distance / C;
        
        // Проверка сходимости 
        if (std::abs(new_delta - delta) < tolerance) {
            return new_delta;
        }
        
        delta = new_delta;
    }
    
    std::cerr << "Предупреждение: поправка за световое время не сошлась после " 
              << max_iter << " итераций" << std::endl;
    return delta;
}

// Создание модели системы
std::vector<Object> create_model_system() {
    std::vector<Object> objects;
    
    const double EARTH_MASS = 5.972e24;
    const double JUPITER_MASS = 1.898e27;
    
    // Kobresia (астероид)
    Object kobresia("Kobresia", 3.0e16);
    kobresia.position = Vec3d(-1.247231001936561E+08, 3.501040899062184E+08, 1.544614456531207E+08) * 1000.0;
    kobresia.velocity = Vec3d(-1.663615216546555E+01, -4.632952969649452E+00, -3.889776207120005E+00) * 1000.0;
    
    // Солнце
    Object sun("Sun", SUN_MASS);
    sun.position = Vec3d(-1.359633615694579E+06,  1.163993750658928E+05,  8.372652597099889E+04) * 1000.0;
    sun.velocity = Vec3d(-3.380122006909895E-04, -1.447967375853807E-02, -6.132577922009313E-03) * 1000.0;
    
    // Юпитер
    Object jupiter("Jupiter", JUPITER_MASS);
    jupiter.position = Vec3d(7.381426304618969E+08,  5.029757079706200E+07,  3.592074173013674E+06) * 1000.0;
    jupiter.velocity = Vec3d(-9.927319384212236E-01,  1.253707948797421E+01,  5.397847549901059E+00) * 1000.0;
    
    // Земля
    Object earth("Earth", EARTH_MASS);
    earth.position = Vec3d(1.475204020619656E+08,  1.570888911282263E+07,  6.842172695828106E+06) * 1000.0;
    earth.velocity = Vec3d(-3.872927841529505E+00,  2.704280911152226E+01,  1.172423083084365E+01) * 1000.0;
    
    objects.push_back(kobresia);
    objects.push_back(sun);
    objects.push_back(jupiter);
    objects.push_back(earth);
    
    return objects;
}

// Интегрирование системы
void integrate_system(const std::vector<Object>& objects, double t_start, double t_end, 
                     double dt, std::vector<SystemState>& states) {
    SystemState current_state;
    init_state(objects, current_state);
    current_state.time = t_start;
    
    states.clear();
    states.push_back(current_state);
    
    int total_steps = static_cast<int>((t_end - t_start) / dt);
    
    for (int step = 1; step <= total_steps; ++step) {
        dopri5_step(current_state, objects, dt);
        states.push_back(current_state);
    }
    
    std::cout << "Интегрирование завершено. Шагов: " << total_steps 
              << ", состояний сохранено: " << states.size() << std::endl;
}

// Проверка точности интерполяции
void check_interpolation_accuracy(const std::vector<SystemState>& states,
                                 const std::vector<Observation>& observations) {
    if (states.size() < 10) return;
    
    // Создание траекторий для проверки
    std::vector<Vec3d> kob_trajectory, earth_trajectory;
    std::vector<double> times;
    
    for (const auto& state : states) {
        kob_trajectory.push_back(state.positions[0]);    // Kobresia
        earth_trajectory.push_back(state.positions[3]);  // Земля
        times.push_back(state.time);
    }
    
    try {
        CubicHermiteInterpolator kob_interp(kob_trajectory, times);
        
        // 1. Проверка на исходных точках
        double max_error = 0.0;
        double avg_error = 0.0;
        int check_count = 0;
        
        for (size_t i = 1; i < states.size() - 2; i += std::max(1, (int)states.size()/100)) {
            double t = times[i];
            Vec3d original = kob_trajectory[i];
            Vec3d interpolated = kob_interp.interpolate(t);
            
            double error = (original - interpolated).length();
            avg_error += error;
            max_error = std::max(max_error, error);
            check_count++;
        }
        
        avg_error /= check_count;
        std::cout << "\nПроверка точности интерполяции:" << std::endl;
        std::cout << "  Средняя ошибка: " << avg_error << " м" << std::endl;
        std::cout << "  Максимальная ошибка: " << max_error << " м" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка при проверке точности интерполяции: " << e.what() << std::endl;
    }
}

// Вспомогательная структура для хранения результатов вычислений
struct ObservationResult {
    double time_jd;
    double ra_computed;
    double dec_computed;
    double dra_arcsec;
    double ddec_arcsec;
    double light_time;
    double deflection_rad;
    double aberration_rad;
    double impact_param;
    double earth_velocity;
};

// Сравнение с наблюдениями
void compare_with_observations(const std::vector<SystemState>& states, 
                              std::vector<Observation>& observations,
                              std::vector<ObservationResult>& results) {
    if (observations.empty()) {
        std::cout << "Нет наблюдений для сравнения." << std::endl;
        return;
    }
    
    // Очищаем результаты
    results.clear();
    
    // Создание траекторий для интерполяции
    std::vector<Vec3d> kob_trajectory, earth_trajectory, sun_trajectory, jupiter_trajectory;
    std::vector<Vec3d> earth_velocity_trajectory;
    std::vector<double> times;
    
    for (const auto& state : states) {
        kob_trajectory.push_back(state.positions[0]);    // Kobresia
        earth_trajectory.push_back(state.positions[3]);  // Земля
        sun_trajectory.push_back(state.positions[1]);    // Солнце
        jupiter_trajectory.push_back(state.positions[2]); // Юпитер
        earth_velocity_trajectory.push_back(state.velocities[3]); // Скорость Земли
        times.push_back(state.time);
    }
    
    try {
        CubicHermiteInterpolator kob_interp(kob_trajectory, times);
        CubicHermiteInterpolator earth_interp(earth_trajectory, times);
        CubicHermiteInterpolator sun_interp(sun_trajectory, times);
        CubicHermiteInterpolator jupiter_interp(jupiter_trajectory, times);
        CubicHermiteInterpolator earth_vel_interp(earth_velocity_trajectory, times);
        
        // Файлы для вывода
        std::ofstream residuals_file("residuals.txt");
        std::ofstream details_file("observation_details.txt");
        std::ofstream corrections_file("relativistic_corrections.txt");
        
        residuals_file << std::setprecision(12);
        details_file << std::setprecision(12);
        corrections_file << std::setprecision(12);
        
        residuals_file << "# time_JD dra_arcsec ddec_arcsec light_time_s deflection_rad aberration_rad\n";
        details_file << "# time_JD observed_RA_deg observed_DEC_deg computed_RA_deg computed_DEC_deg "
                     << "dra_deg ddec_deg light_time_s\n";
        corrections_file << "# time_JD deflection_arcsec aberration_arcsec impact_param_m v_earth_m_s\n";
        
        double sum_sq_ra = 0.0, sum_sq_dec = 0.0;
        int n = observations.size();
        
        const double JUPITER_MASS = 1.898e27;
        
        for (int i = 0; i < n; ++i) {
            Observation& obs = observations[i];
            
            // 1. Получаем положение и скорость Земли в момент наблюдения
            Vec3d earth_pos = earth_interp.interpolate(obs.time_seconds);
            Vec3d earth_vel = earth_vel_interp.interpolate(obs.time_seconds);
            Vec3d sun_pos = sun_interp.interpolate(obs.time_seconds);
            Vec3d jupiter_pos = jupiter_interp.interpolate(obs.time_seconds);
            
            // 2. Вычисляем гравитационный потенциал в точке Земли
            double Ue_sun = earth_gravitational_potential(earth_pos, sun_pos);
            
            Vec3d d_jup = jupiter_pos - earth_pos;
            double r_jup = d_jup.length();
            double Ue_jupiter = 0.0;
            if (r_jup > 1e-10) {
                Ue_jupiter = G * JUPITER_MASS / r_jup;
            }
            
            double Ue = Ue_sun + Ue_jupiter;
            
            // 3. Преобразуем положение и скорость обсерватории в BCRS 
            PV obs_bary = gcrs_to_bcrs_relativistic(
                obs.obs_position_gcrs,
                obs.obs_velocity_gcrs,
                earth_pos,
                earth_vel,
                Ue
            );
            
            obs.obs_position_bcrs = obs_bary.r;
            obs.obs_velocity_bcrs = obs_bary.v;
            
            // 4. Вычисляем поправку за световое время
            double light_time = compute_light_time_correction(
                obs.time_seconds, obs.obs_position_bcrs, kob_interp);
            
            double t_emit = obs.time_seconds - light_time;
            
            // 5. Положения тел в момент испускания/приема
            Vec3d ast_pos = kob_interp.interpolate(t_emit);
            
            // 6. Топоцентрический вектор и направление
            Vec3d topocentric = ast_pos - obs.obs_position_bcrs;
            Vec3d direction = topocentric.normalized();
            
            // 7. Релятивистские поправки
            Vec3d direction_corrected = direction;
            
            if (APPLY_GRAVITATIONAL_DEFLECTION) {
                direction_corrected = apply_gravitational_deflection(
                    direction_corrected, obs.obs_position_bcrs, ast_pos, sun_pos);
            }
            
            if (APPLY_ABERRATION) {
                direction_corrected = apply_aberration(direction_corrected, obs.obs_velocity_bcrs);
            }
            
            // 8. Вычисленные координаты
            auto comp_ang = cartesian_to_ra_dec(direction_corrected);
            double ra_computed = comp_ang.first;
            double dec_computed = comp_ang.second;
            
            // 9. Невязки
            double dra = obs.ra_observed - ra_computed;
            double ddec = obs.dec_observed - dec_computed;
            
            // Нормализация разности RA
            while (dra > PI) dra -= 2.0 * PI;
            while (dra < -PI) dra += 2.0 * PI;
            
            double dra_arcsec = dra * RAD_TO_ARCSEC;
            double ddec_arcsec = ddec * RAD_TO_ARCSEC;
            
            // 10. Величины поправок
            double deflection_rad = 0.0;
            if (APPLY_GRAVITATIONAL_DEFLECTION) {
                Vec3d deflected = apply_gravitational_deflection(
                    direction, obs.obs_position_bcrs, ast_pos, sun_pos);
                deflection_rad = std::acos(std::max(-1.0, std::min(1.0, direction * deflected)));
            }
            
            double aberration_rad = 0.0;
            if (APPLY_ABERRATION) {
                Vec3d aberrated = apply_aberration(direction, obs.obs_velocity_bcrs);
                aberration_rad = std::acos(std::max(-1.0, std::min(1.0, direction * aberrated)));
            }
            
            double impact_param = compute_impact_parameter(obs.obs_position_bcrs, ast_pos, sun_pos);
            
            // 11. Создаем результат
            ObservationResult result;
            result.time_jd = obs.time_jd;
            result.ra_computed = ra_computed;
            result.dec_computed = dec_computed;
            result.dra_arcsec = dra_arcsec;
            result.ddec_arcsec = ddec_arcsec;
            result.light_time = light_time;
            result.deflection_rad = deflection_rad;
            result.aberration_rad = aberration_rad;
            result.impact_param = impact_param;
            result.earth_velocity = earth_vel.length();
            
            results.push_back(result);
            
            // 12. Запись в файлы
            residuals_file << obs.time_jd << " " << dra_arcsec << " " << ddec_arcsec << " "
                          << light_time << " " << deflection_rad << " " << aberration_rad << "\n";
            
            details_file << obs.time_jd << " "
                        << obs.ra_observed * RAD_TO_DEG << " " << obs.dec_observed * RAD_TO_DEG << " "
                        << ra_computed * RAD_TO_DEG << " " << dec_computed * RAD_TO_DEG << " "
                        << dra * RAD_TO_DEG << " " << ddec * RAD_TO_DEG << " "
                        << light_time << "\n";
            
            corrections_file << obs.time_jd << " "
                            << deflection_rad * RAD_TO_ARCSEC << " " 
                            << aberration_rad * RAD_TO_ARCSEC << " "
                            << impact_param << " " << earth_vel.length() << "\n";
            
            sum_sq_ra += dra_arcsec * dra_arcsec;
            sum_sq_dec += ddec_arcsec * ddec_arcsec;
        }
        
        residuals_file.close();
        details_file.close();
        corrections_file.close();
        
        if (n > 0) {
            double rms_ra = std::sqrt(sum_sq_ra / n);
            double rms_dec = std::sqrt(sum_sq_dec / n);
            
            std::cout << "\n=== РЕЗУЛЬТАТЫ ===" << std::endl;
            std::cout << "Обработано наблюдений: " << n << std::endl;
            std::cout << "RMS невязки (угл. сек):" << std::endl;
            std::cout << "  RA:  " << rms_ra << " угл. сек" << std::endl;
            std::cout << "  DEC: " << rms_dec << " угл. сек" << std::endl;
            std::cout << "RMS невязки (градусы):" << std::endl;
            std::cout << "  RA:  " << rms_ra * ARCSEC_TO_RAD * RAD_TO_DEG << " градусов" << std::endl;
            std::cout << "  DEC: " << rms_dec * ARCSEC_TO_RAD * RAD_TO_DEG << " градусов" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка при создании интерполяторов: " << e.what() << std::endl;
    }
}

// Основная функция
int main() {
    // Параметры интегрирования
    double start_jd = 2459852.5;
    double duration_days = 9.0;
    double dt = 60.0; 
    
    double t_start = jd_to_seconds(start_jd);
    double t_end = t_start + duration_days * 86400.0;
    
    std::cout << "Интегрирование траекторий..." << std::endl;
    std::cout << "  Начальное время: " << start_jd << " JD" << std::endl;
    std::cout << "  Длительность: " << duration_days << " дней" << std::endl;
    std::cout << "  Шаг интегрирования: " << dt << " секунд" << std::endl;
    
    // Создание системы
    std::vector<Object> system = create_model_system();
    
    // Интегрирование
    std::vector<SystemState> states;
    integrate_system(system, t_start, t_end, dt, states);
    
    std::cout << "\nЧтение наблюдений из файла output.txt..." << std::endl;
    std::vector<Observation> observations = read_observed_data("output.txt");
    
    if (observations.empty()) {
        std::cerr << "Нет наблюдений для обработки. Завершение программы." << std::endl;
        return 1;
    }
    
    check_interpolation_accuracy(states, observations);
    
    // Сравнение с наблюдениями
    std::vector<ObservationResult> results;
    if (!observations.empty()) {
        std::cout << "\nСравнение с наблюдениями..." << std::endl;
        compare_with_observations(states, observations, results);
    }
    
    // Сохранение траектории для анализа
    std::ofstream traj_file("trajectory.txt");
    traj_file << std::setprecision(12);
    traj_file << "# time_seconds time_JD x_kob_m y_kob_m z_kob_m x_earth_m y_earth_m z_earth_m\n";
    
    for (const auto& state : states) {
        if (state.time >= t_start && state.time <= t_end) {
            traj_file << state.time << " " << seconds_to_jd(state.time) << " "
                     << state.positions[0].x << " " << state.positions[0].y << " " << state.positions[0].z << " "
                     << state.positions[3].x << " " << state.positions[3].y << " " << state.positions[3].z << "\n";
        }
    }
    traj_file.close();
    
    std::cout << "\nРезультаты сохранены в файлы:" << std::endl;
    std::cout << "  trajectory.txt - траектория астероида и Земли" << std::endl;
    std::cout << "  residuals.txt - невязки наблюдений" << std::endl;
    std::cout << "  observation_details.txt - подробные данные наблюдений" << std::endl;
    std::cout << "  relativistic_corrections.txt - релятивистские поправки" << std::endl;
    
    return 0;
}
