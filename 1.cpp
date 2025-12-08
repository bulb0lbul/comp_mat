// g++ -std=c++17 -O2 1.cpp \
   -I/opt/homebrew/include/eigen3 \
   -I/Users/yanaprokopovich/Downloads/sofa/20231011/c/src \
   -L/Users/yanaprokopovich/Downloads/sofa/20231011/c/src -lsofa \
   -o 1
// ./1 initial_state.txt ephem_folder filtered.csv  

#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <functional>
#include <iostream>  

#include "sofa.h"
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::VectorXd;

static constexpr double DAYSEC = 86400.0;
static constexpr double C_KM_S = 299792.458;
static constexpr double AU_KM = 149597870.7;
static constexpr double REQ = 6378.140; // equatorial Earth radius km
static constexpr double GM_SUN = 1.32712440018e11; // Added for Shapiro delay

// default GM map (km^3/s^2)
static const std::map<std::string, double> DEFAULT_GM = {
    {"sun", 1.32712440018e11},
    {"mercury", 22031.868551},
    {"venus", 324858.592000},
    {"earth", 398600.435507},
    {"moon", 4902.800118},
    {"mars", 42828.375816},
    {"jupiter", 126712764.100000},
    {"saturn", 37940584.841800}
};

// -------------------- Type definitions --------------------
typedef Eigen::VectorXd state_type; // [x,y,z,vx,vy,vz] - теперь Eigen::VectorXd

// -------------------- Forward declarations --------------------
class CubicSpline1D;
struct SplineTraj;
class BodyEphemeris;

// -------------------- simple 1D natural cubic spline --------------------
class CubicSpline1D {
public:
    void build(const std::vector<double>& x_, const std::vector<double>& y_) {
        x = x_; y = y_;
        int n = (int)x.size();
        a = y;
        if (n < 2) { b.assign(std::max(0,n-1),0.0); c.assign(n,0.0); d.assign(std::max(0,n-1),0.0); return; }
        std::vector<double> h(n-1);
        for (int i=0;i<n-1;++i) h[i] = x[i+1]-x[i];
        std::vector<double> alpha(n,0.0);
        for (int i=1;i<n-1;++i)
            alpha[i] = 3.0*( (a[i+1]-a[i])/h[i] - (a[i]-a[i-1])/h[i-1] );
        std::vector<double> l(n), mu(n), z(n);
        l[0]=1.0; mu[0]=z[0]=0.0;
        for (int i=1;i<n-1;++i) {
            l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1];
            mu[i] = h[i]/l[i];
            z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i];
        }
        l[n-1]=1.0; z[n-1]=0.0; c.assign(n,0.0);
        for (int j=n-2;j>=0;--j) {
            c[j] = z[j] - mu[j]*c[j+1];
        }
        b.assign(n-1,0.0); d.assign(n-1,0.0);
        for (int i=0;i<n-1;++i) {
            b[i] = (a[i+1]-a[i])/h[i] - h[i]*(c[i+1]+2.0*c[i])/3.0;
            d[i] = (c[i+1]-c[i])/(3.0*h[i]);
        }
    }

    double eval(double xq) const {
        int n = (int)x.size();
        if (n==0) return 0.0;
        if (n==1) return a[0];
        if (xq <= x.front()) return a.front();
        if (xq >= x.back()) return a.back();
        int lo=0, hi=n-1;
        while (hi-lo>1) {
            int mid=(lo+hi)/2;
            if (x[mid] <= xq) lo = mid; else hi = mid;
        }
        double dx = xq - x[lo];
        return a[lo] + b[lo]*dx + c[lo]*dx*dx + d[lo]*dx*dx*dx;
    }

    bool empty() const { return x.empty(); }

private:
    std::vector<double> x, a, b, c, d, y;
};

// -------------------- Ephemeris container with cubic splines --------------------
struct EphemPoint {
    double jd;
    Vector3d pos; // km (ICRS orientation)
    std::optional<Vector3d> vel; // km/s
};

class BodyEphemeris {
public:
    std::string name;
    std::vector<EphemPoint> pts;
    bool has_vel = false;
    CubicSpline1D sx, sy, sz;
    CubicSpline1D svx, svy, svz;
    bool splines_built = false;

    BodyEphemeris() {}
    BodyEphemeris(const std::string &n) : name(n) {}

    bool load_csv(const std::string &path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::string t = line;
            auto ltrim = [](std::string &s){ s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch){ return !std::isspace(ch); })); };
            auto rtrim = [](std::string &s){ s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch){ return !std::isspace(ch); }).base(), s.end()); };
            ltrim(t); rtrim(t);
            if (t.empty()) continue;
            if (t.rfind("JDTDB",0) == 0) continue;
            if (t.rfind("$$SOE",0) == 0) continue;
            if (t.rfind("$$EOE",0) == 0) break;
            if (t[0] == '#' || t[0] == '*') continue;
            for (char &c : t) if (c == ',') c = ' ';
            std::stringstream ss(t);
            double jd,x,y,z;
            if (!(ss >> jd >> x >> y >> z)) {
                std::vector<double> nums; std::string token; std::stringstream ss2(t);
                while (ss2 >> token) {
                    try { size_t idx=0; double v = std::stod(token,&idx); nums.push_back(v);} catch(...) { }
                }
                if (nums.size() >= 4) {
                    jd = nums[0]; x = nums[1]; y = nums[2]; z = nums[3];
                    EphemPoint p; p.jd = jd; p.pos = Vector3d(x,y,z); p.vel = std::nullopt; pts.push_back(p); continue;
                } else continue;
            }
            double vx,vy,vz; bool hasv=false;
            if (ss >> vx >> vy >> vz) hasv = true;
            EphemPoint p; p.jd = jd; p.pos = Vector3d(x,y,z);
            if (hasv) { p.vel = Vector3d(vx,vy,vz); has_vel = true; } else p.vel = std::nullopt;
            pts.push_back(p);
        }
        if (pts.empty()) return false;
        std::sort(pts.begin(), pts.end(), [](const EphemPoint &a, const EphemPoint &b){ return a.jd < b.jd; });
        build_splines();
        return true;
    }

    void build_splines() {
        if (pts.size() < 2) { splines_built = false; return; }
        std::vector<double> jds, xs, ys, zs, vxs, vys, vzs;
        for (const auto &p : pts) {
            jds.push_back(p.jd);
            xs.push_back(p.pos.x());
            ys.push_back(p.pos.y());
            zs.push_back(p.pos.z());
            if (p.vel) { vxs.push_back(p.vel->x()); vys.push_back(p.vel->y()); vzs.push_back(p.vel->z()); }
        }
        sx.build(jds, xs); sy.build(jds, ys); sz.build(jds, zs);
        if (has_vel && vxs.size()==jds.size()) {
            svx.build(jds, vxs); svy.build(jds, vys); svz.build(jds, vzs);
        }
        splines_built = true;
    }

    Vector3d position_at(double jd) const {
        if (!splines_built || pts.empty()) {
            if (pts.empty()) return Vector3d::Zero();
            if (jd <= pts.front().jd) return pts.front().pos;
            if (jd >= pts.back().jd) return pts.back().pos;
        }
        if (splines_built) {
            return Vector3d(sx.eval(jd), sy.eval(jd), sz.eval(jd));
        } else {
            if (jd <= pts.front().jd) return pts.front().pos;
            if (jd >= pts.back().jd) return pts.back().pos;
            size_t lo=0, hi=pts.size()-1;
            while (hi-lo>1) { size_t mid=(lo+hi)/2; if (pts[mid].jd <= jd) lo=mid; else hi=mid; }
            double t = (jd - pts[lo].jd) / (pts[hi].jd - pts[lo].jd);
            return (1.0-t)*pts[lo].pos + t*pts[hi].pos;
        }
    }

    std::optional<Vector3d> velocity_at(double jd) const {
        if (!has_vel) return std::nullopt;
        if (!splines_built) {
            if (jd <= pts.front().jd) return pts.front().vel;
            if (jd >= pts.back().jd) return pts.back().vel;
            size_t lo=0, hi=pts.size()-1;
            while (hi-lo>1) { size_t mid=(lo+hi)/2; 
            if (pts[mid].jd <= jd) lo=mid; else hi=mid; }
            double t = (jd - pts[lo].jd) / (pts[hi].jd - pts[lo].jd);
            return (1.0-t)*pts[lo].vel.value() + t*pts[hi].vel.value();
        } else {
            return Vector3d(svx.eval(jd), svy.eval(jd), svz.eval(jd));
        }
    }

    size_t size() const { return pts.size(); }
};

// -------------------- trajectory spline built from recorder --------------------
struct SplineTraj {
    CubicSpline1D sx, sy, sz;
    std::vector<double> jd;
    bool built=false;
    void build_from_record(const std::vector<double> &times_seconds, const std::vector<state_type> &states, double jd0) {
        jd.clear();
        std::vector<double> xs, ys, zs;
        size_t n = times_seconds.size();
        jd.reserve(n); xs.reserve(n); ys.reserve(n); zs.reserve(n);
        for (size_t i=0;i<n;++i){
            double j = jd0 + times_seconds[i]/DAYSEC;
            jd.push_back(j);
            const state_type &s = states[i];
            xs.push_back(s(0)); ys.push_back(s(1)); zs.push_back(s(2));
        }
        sx.build(jd,xs); sy.build(jd,ys); sz.build(jd,zs);
        built = true;
    }
    Vector3d position_at(double jdq) const {
        if (!built || jd.empty()) return Vector3d::Zero();
        if (jdq <= jd.front()) return Vector3d(sx.eval(jd.front()), sy.eval(jd.front()), sz.eval(jd.front()));
        if (jdq >= jd.back()) return Vector3d(sx.eval(jd.back()), sy.eval(jd.back()), sz.eval(jd.back()));
        return Vector3d(sx.eval(jdq), sy.eval(jdq), sz.eval(jdq));
    }
};

// -------------------- Relativistic acceleration functions --------------------
Vector3d acc_relativistic_schwarzschild(const Vector3d &r, const Vector3d &v, double mu) {
    const double c2 = C_KM_S * C_KM_S;
    double rnorm = r.norm();
    
    if (rnorm < 1e-12) return Vector3d::Zero();
    
    double r2 = rnorm * rnorm;
    double r3 = r2 * rnorm;
    double v2 = v.squaredNorm();
    double rdotv = r.dot(v);
    
    // Шварцшильдовская метрика (Post-Newtonian 1PN)
    Vector3d a_pn = (mu / (c2 * r3)) * (
        (4.0 * mu / rnorm - v2) * r + 4.0 * rdotv * v
    );
    
    return a_pn;
}

Vector3d acc_lense_thirring_sun(const Vector3d &r, const Vector3d &v, double mu_sun) {
    const double c2 = C_KM_S * C_KM_S;
    const double S_sun = 1.92e41;  // Угловой момент Солнца (kg·km²/s)
    const double G = 6.67430e-20;  // Гравитационная постоянная (km³/kg/s²)
    
    double rnorm = r.norm();
    if (rnorm < 1e-12) return Vector3d::Zero();
    
    // Направление оси вращения Солнца 
    Vector3d omega_sun(0.0, 0.0, 2.865e-6);  // рад/сек
    
    // Формула Лензе-Тирринга
    Vector3d a_lt = (2.0 * G / c2) * (
        (3.0 * r.dot(omega_sun) * r.cross(v) / (rnorm * rnorm * rnorm * rnorm)) +
        (v.cross(omega_sun) / (rnorm * rnorm * rnorm))
    );
    
    return a_lt * S_sun;
}

// -------------------- Solve light-time with Shapiro delay --------------------
std::pair<double, Vector3d> solve_light_time_relativistic(
    double obs_jd, const Vector3d& obs_pos, const SplineTraj& traj, 
    const Vector3d& obs_vel = Vector3d::Zero(), bool include_shapiro = false) {
    
    const double c = C_KM_S;
    double delta = 0.0;
    Vector3d r_emit;
    
    for (int iter = 0; iter < 20; ++iter) {
        double emit_jd = obs_jd - delta / DAYSEC;
        r_emit = traj.position_at(emit_jd);
        
        // Учет движения наблюдателя за время delta
        Vector3d obs_pos_at_emit = obs_pos - obs_vel * delta;
        
        Vector3d dr = r_emit - obs_pos_at_emit;
        double dist = dr.norm();
        
        // Релятивистская поправка 
        double shapiro_delay = 0.0;
        if (include_shapiro) {
            // Вычисление задержки Шапиро
            double r1 = obs_pos_at_emit.norm();
            double r2 = r_emit.norm();
            double dr_dot = dr.norm();
            shapiro_delay = (2.0 * GM_SUN / (c * c * c)) * log((r1 + r2 + dr_dot) / (r1 + r2 - dr_dot));
        }
        
        double new_delta = dist / c + shapiro_delay;
        
        if (std::abs(new_delta - delta) < 1e-12) {
            return {new_delta, r_emit};
        }
        delta = new_delta;
    }
    return {delta, r_emit};
}

// -------------------- Integrator context and acceleration --------------------
struct IntegratorContext {
    double jd0 = 0.0;
    std::map<std::string, BodyEphemeris> ephems;
    std::map<std::string, double> gm;
    bool ephems_are_barycentric = false;
    
    // --- НОВЫЕ ПОЛЯ ДЛЯ РЕЛЯТИВИСТСКИХ ЭФФЕКТОВ ---
    bool include_relativistic = false;          // Релятивистские поправки
    bool include_lense_thirring = false;        // Эффект Лензе-Тирринга
    bool include_j2_relativistic = false;       // J2 + релятивизм для планет
    bool include_shapiro = false;               // Задержка Шапиро
    
    // Параметры вращения тел
    std::map<std::string, Vector3d> body_angular_momentum;
};

// вычисляет полное ускорение
Vector3d acc_barycentric(double jd, const Vector3d &r, const Vector3d &v, const IntegratorContext &ctx) {
    Vector3d a = Vector3d::Zero();
    
    // 1. Ускорение от Солнца (ньютоновское + релятивистское)
    auto it_sun = ctx.gm.find("sun");
    if (it_sun == ctx.gm.end()) throw std::runtime_error("GM for sun missing.");
    double mu_sun = it_sun->second;
    
    Vector3d r_sun = Vector3d::Zero();
    Vector3d v_sun = Vector3d::Zero();
    bool has_sun_vel = false;
    
    if (ctx.ephems.count("sun")) {
        r_sun = ctx.ephems.at("sun").position_at(jd);
        auto v_sun_opt = ctx.ephems.at("sun").velocity_at(jd);
        if (v_sun_opt) {
            v_sun = *v_sun_opt;
            has_sun_vel = true;
        }
    }
    Vector3d dr_sun = r - r_sun;
    double dsun = dr_sun.norm();
    
    if (dsun > 1e-12) {
        // Ньютоновское ускорение
        a += -mu_sun * dr_sun / (dsun * dsun * dsun);
        // Релятивистские поправки от Солнца
        if (ctx.include_relativistic) {
            // Используем относительную скорость для релятивистских поправок
            Vector3d v_rel = has_sun_vel ? (v - v_sun) : v;
            // Шварцшильдовская метрика (Post-Newtonian 1PN)
            a += acc_relativistic_schwarzschild(dr_sun, v_rel, mu_sun);
            // Эффект Лензе-Тирринга (увлечение инерциальных систем)
            if (ctx.include_lense_thirring) {
                a += acc_lense_thirring_sun(dr_sun, v_rel, mu_sun);
            }
        }
    }
    
    // 2. Ускорение от других тел
    for (const auto &p : ctx.ephems) {
        const std::string name = p.first;
        if (name == "sun") continue;
        auto itgm = ctx.gm.find(name);
        if (itgm == ctx.gm.end()) continue;
        double mu = itgm->second;
        Vector3d r_body = p.second.position_at(jd);
        Vector3d dr = r - r_body;
        double d = dr.norm();
        if (d > 1e-12) {
            // Ньютоновское ускорение
            a += -mu * dr / (d * d * d);
            // Релятивистские поправки от планет 
            if (ctx.include_relativistic && ctx.include_j2_relativistic) {
                Vector3d v_body = Vector3d::Zero();
                auto v_body_opt = p.second.velocity_at(jd);
                if (v_body_opt) v_body = *v_body_opt;
                Vector3d v_rel = v - v_body;
                a += acc_relativistic_schwarzschild(dr, v_rel, mu);
            }
        }
    }
    return a;
}

struct RHS {
    const IntegratorContext *pctx;
    double jd0;
    RHS() : pctx(nullptr), jd0(0.0) {}
    RHS(const IntegratorContext &ctx, double _jd0) : pctx(&ctx), jd0(_jd0) {}
    void operator()(double t, const state_type &x, state_type &dxdt) const {
        double jd = jd0 + t / DAYSEC;
        Vector3d r(x(0), x(1), x(2));
        Vector3d v(x(3), x(4), x(5));
        Vector3d a = acc_barycentric(jd, r, v, *pctx);
        dxdt.resize(6);
        dxdt(0) = v(0); dxdt(1) = v(1); dxdt(2) = v(2);
        dxdt(3) = a(0); dxdt(4) = a(1); dxdt(5) = a(2);
    }
};

class DormandPrince5 {
private:
    // Коэффициенты метода Дормана-Принса
    static constexpr double a2 = 1.0/5.0;
    static constexpr double a3 = 3.0/10.0;
    static constexpr double a4 = 4.0/5.0;
    static constexpr double a5 = 8.0/9.0;
    static constexpr double a6 = 1.0;
    static constexpr double a7 = 1.0;
    
    static constexpr double b21 = 1.0/5.0;
    static constexpr double b31 = 3.0/40.0;
    static constexpr double b32 = 9.0/40.0;
    static constexpr double b41 = 44.0/45.0;
    static constexpr double b42 = -56.0/15.0;
    static constexpr double b43 = 32.0/9.0;
    static constexpr double b51 = 19372.0/6561.0;
    static constexpr double b52 = -25360.0/2187.0;
    static constexpr double b53 = 64448.0/6561.0;
    static constexpr double b54 = -212.0/729.0;
    static constexpr double b61 = 9017.0/3168.0;
    static constexpr double b62 = -355.0/33.0;
    static constexpr double b63 = 46732.0/5247.0;
    static constexpr double b64 = 49.0/176.0;
    static constexpr double b65 = -5103.0/18656.0;
    static constexpr double b71 = 35.0/384.0;
    static constexpr double b72 = 0.0;
    static constexpr double b73 = 500.0/1113.0;
    static constexpr double b74 = 125.0/192.0;
    static constexpr double b75 = -2187.0/6784.0;
    static constexpr double b76 = 11.0/84.0;
    
    // Коэффициенты для 5-го порядка (основной метод)
    static constexpr double c1 = 35.0/384.0;
    static constexpr double c2 = 0.0;
    static constexpr double c3 = 500.0/1113.0;
    static constexpr double c4 = 125.0/192.0;
    static constexpr double c5 = -2187.0/6784.0;
    static constexpr double c6 = 11.0/84.0;
    static constexpr double c7 = 0.0;
    
    // Коэффициенты для 4-го порядка (для оценки ошибки)
    static constexpr double d1 = 5179.0/57600.0;
    static constexpr double d2 = 0.0;
    static constexpr double d3 = 7571.0/16695.0;
    static constexpr double d4 = 393.0/640.0;
    static constexpr double d5 = -92097.0/339200.0;
    static constexpr double d6 = 187.0/2100.0;
    static constexpr double d7 = 1.0/40.0;
    
public:
    // Выполняет один шаг метода Дормана-Принса, возвращает оценку ошибки и новый шаг
    template<typename System>
    static std::pair<double, state_type> step(
        System& system, double t, const state_type& y, double h) {
        
        // Вычисляем коэффициенты k1-k7
        state_type k1(6), k2(6), k3(6), k4(6), k5(6), k6(6), k7(6);
        
        // k1 = f(t, y)
        system(t, y, k1);
        
        // k2 = f(t + a2*h, y + h*(b21*k1))
        state_type y_temp = y + h * b21 * k1;
        system(t + a2*h, y_temp, k2);
        
        // k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        y_temp = y + h * (b31*k1 + b32*k2);
        system(t + a3*h, y_temp, k3);
        
        // k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        y_temp = y + h * (b41*k1 + b42*k2 + b43*k3);
        system(t + a4*h, y_temp, k4);
        
        // k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        y_temp = y + h * (b51*k1 + b52*k2 + b53*k3 + b54*k4);
        system(t + a5*h, y_temp, k5);
        
        // k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        y_temp = y + h * (b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5);
        system(t + a6*h, y_temp, k6);
        
        // k7 = f(t + a7*h, y + h*(b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6))
        y_temp = y + h * (b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6);
        system(t + a7*h, y_temp, k7);
        
        // Вычисляем приближение 5-го порядка
        state_type y5 = y + h * (c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6 + c7*k7);
        
        // Вычисляем приближение 4-го порядка (для оценки ошибки)
        state_type y4 = y + h * (d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*k7);
        
        // Оценка ошибки (максимальная относительная ошибка по компонентам)
        state_type error_vec = y5 - y4;
        double error = 0.0;
        for (int i = 0; i < 6; ++i) {
            double scale = std::max(std::abs(y(i)), 1.0);
            double rel_error = std::abs(error_vec(i)) / scale;
            if (rel_error > error) error = rel_error;
        }
        
        return {error, y5};
    }
};

class AdaptiveDOPRI5 {
private:
    double atol_;  // Абсолютный допуск
    double rtol_;  // Относительный допуск
    double hmin_;  // Минимальный шаг
    double hmax_;  // Максимальный шаг
    double fac_;   // Фактор безопасности для изменения шага
    double facmax_; // Максимальный коэффициент увеличения шага
    
public:
    AdaptiveDOPRI5(double atol = 1e-10, double rtol = 1e-10, double hmin = 1e-6, double hmax = 1e4, double fac = 0.9, double facmax = 5.0): atol_(atol), rtol_(rtol), hmin_(hmin), hmax_(hmax), fac_(fac), facmax_(facmax) {}
    // Интегрирует систему ОДУ от t0 до t1 с начальным шагом h
    template<typename System, typename Observer>
    void integrate(System& system, state_type& y, double t0, double t1, double h, Observer observer) {  
        double t = t0;
        int steps = 0;
        int rejected_steps = 0;
        // Вызываем наблюдателя для начального состояния
        observer(y, t);
        while (t < t1) {
            // Ограничиваем шаг, чтобы не выйти за t1
            if (t + h > t1) {
                h = t1 - t;
            }
            // Пытаемся сделать шаг
            auto [error, y_new] = DormandPrince5::step(system, t, y, h);
            // Вычисляем допустимую ошибку
            double tol = rtol_ * y.norm() + atol_;
            // Если ошибка приемлема, принимаем шаг
            if (error <= tol) {
                t += h;
                y = y_new;
                steps++;
                // Вызываем наблюдателя
                observer(y, t);
                // Увеличиваем шаг для следующей итерации
                if (error > 0) {
                    double factor = fac_ * std::pow(tol / error, 0.2);
                    factor = std::min(factor, facmax_);
                    h *= factor;
                } else {
                    h *= facmax_;
                }
            } else {
                // Шаг отвергнут, уменьшаем шаг
                rejected_steps++;
                double factor = fac_ * std::pow(tol / error, 0.25);
                h *= std::max(0.1, factor);
            }
            // Ограничиваем шаг
            if (h < hmin_) {
                std::cerr << "Warning: step size below minimum at t = " << t << std::endl;
                h = hmin_;
            }
            if (h > hmax_) {
                h = hmax_;
            }
            // Защита от бесконечного цикла
            if (steps > 1000000) {
                std::cerr << "Warning: maximum number of steps reached" << std::endl;
                break;
            }
        }
        std::cerr << "Integration complete: " << steps << " steps, "  << rejected_steps << " rejected steps" << std::endl;
    }
};

bool read_initial_state(const std::string &path, double &jd0, state_type &X0) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        for (char &c : line) if (c == ',') c = ' ';
        std::stringstream ss(line);
        double jd,x,y,z,vx,vy,vz;
        if (ss >> jd >> x >> y >> z >> vx >> vy >> vz) {
            jd0 = jd; 
            X0.resize(6);
            X0(0)=x; X0(1)=y; X0(2)=z; X0(3)=vx; X0(4)=vy; X0(5)=vz;
            return true;
        }
    }
    return false;
}

struct StepRecorder { 
    std::vector<double> times; 
    std::vector<state_type> states; 
    void operator()(const state_type &x , double t) { 
        times.push_back(t); 
        states.push_back(x);
    } 
};

std::pair<double, Vector3d> solve_light_time(double obs_jd, const Vector3d &obs_pos_bary, const SplineTraj &traj, double c_km_s = C_KM_S) {
    Vector3d r_at_obs = traj.position_at(obs_jd);
    double dist0 = (r_at_obs - obs_pos_bary).norm();
    double delta = dist0 / c_km_s;
    for (int it=0; it<60; ++it) {
        double emit_jd = obs_jd - delta / DAYSEC;
        Vector3d r_emit = traj.position_at(emit_jd);
        double new_dist = (r_emit - obs_pos_bary).norm();
        double new_delta = new_dist / c_km_s;
        if (std::abs(new_delta - delta) < 1e-8) { delta = new_delta; return {delta, r_emit}; }
        delta = new_delta;
    }
    double emit_jd = obs_jd - delta / DAYSEC;
    Vector3d r_emit = traj.position_at(emit_jd);
    return {delta, r_emit};
}

struct ObsRecord {
    double jd_tdb; // JD TDB
    std::string time_utc_iso;
    std::string time_tt_iso;
    std::string time_tdb_iso;
    double ra_deg = NAN, dec_deg = NAN;
    double obs_x_km = NAN, obs_y_km = NAN, obs_z_km = NAN; 
    double ut1_utc = 0.0;
    double xp_arcsec = 0.0, yp_arcsec = 0.0; 
    double dtr_seconds = NAN;
};

static inline void split_jd(double jd, double &dj1, double &dj2) {
    dj1 = std::floor(jd);
    dj2 = jd - dj1;
}

bool parse_iso_datetime(const std::string &iso, int &Y, int &M, int &D, int &hh, int &mm, double &ss) {
    if (iso.empty()) return false;
    std::string s = iso;
    s.erase(std::remove(s.begin(), s.end(), '\"'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\''), s.end());
    for (char &c : s) if (c=='T' || c=='t') c=' ';
    std::istringstream ssin(s);
    std::string date, time;
    if (!(ssin >> date)) return false;
    ssin >> time;
    char c1, c2;
    std::stringstream sd(date);
    if (!(sd >> Y >> c1 >> M >> c2 >> D)) return false;
    hh=0; mm=0; ss=0.0;
    if (!time.empty()) {
        std::replace(time.begin(), time.end(), ':', ' ');
        std::stringstream st(time);
        if (!(st >> hh)) return false;
        if (!(st >> mm)) mm = 0;
        std::string secstr; st >> secstr;
        if (secstr.empty()) ss = 0.0;
        else {
            try { ss = std::stod(secstr); } catch(...) { ss = 0.0; }
        }
    }
    return true;
}

double compute_precise_dtr_iers2010(double jd_tt) {
    double T = (jd_tt - 2451545.0) / 36525.0;
    
    // Модель из IERS Conventions 2010
    double dtr = 0.001657 * sin(628.3076*T + 6.2401)
               + 0.000022 * sin(575.3385*T + 4.2970)
               + 0.000014 * sin(1256.6152*T + 6.1969)
               + 0.000005 * sin(606.9777*T + 4.0212)
               + 0.000005 * sin(52.9691*T + 0.4444)
               + 0.000002 * sin(21.3299*T + 5.5431)
               + 0.000010 * T * sin(628.3076*T + 4.2490)
               + 0.000002 * sin(-0.9251*T + 5.5369); 
    
    // Добавляем члены для Луны и планет
    dtr += 0.0000007 * sin(0.19834*T + 0.4533);  // Луна
    dtr += 0.0000003 * sin(0.09192*T + 5.5314);  // Венера
    dtr += 0.0000002 * sin(0.04809*T + 3.1962);  // Марс
    
    return dtr; 
}

bool utciso_to_jdparts_and_tt_ut1(const std::string &iso_utc, double ut1_utc_sec, double dtr_hint_seconds, double &jd_tdb_out, double &tt1, double &tt2, double &ut11, double &ut12) {
    int Y,M,D, hh, mm; double ss;
    if (!parse_iso_datetime(iso_utc, Y,M,D,hh,mm,ss)) return false;
    double utc1, utc2;
    iauDtf2d("UTC", Y, M, D, hh, mm, ss, &utc1, &utc2);
    double tai1, tai2;
    iauUtctai(utc1, utc2, &tai1, &tai2);
    iauTaitt(tai1, tai2, &tt1, &tt2);
    double dtr = 0.0;
    if (std::isfinite(dtr_hint_seconds)) {
        dtr = dtr_hint_seconds;  
    } else {
        // Вычисляем по модели IERS Conventions 2010
        double jd_tt = tt1 + tt2;
        dtr = compute_precise_dtr_iers2010(jd_tt);
    }
    double tdb1, tdb2;
    iauTttdb(tt1, tt2, dtr, &tdb1, &tdb2);
    jd_tdb_out = tdb1 + tdb2;
    double jd_utc = utc1 + utc2;
    double jd_ut1 = jd_utc + ut1_utc_sec / DAYSEC;
    ut11 = std::floor(jd_ut1);
    ut12 = jd_ut1 - ut11;
    return true;
}

bool read_prepared_observations(const std::string &path, std::vector<ObsRecord> &out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string header; if (!std::getline(f, header)) return false;
    std::vector<std::string> cols; std::stringstream sh(header); std::string token;
    while (std::getline(sh, token, ',')) { cols.push_back(token); }
    std::map<std::string, int> idx;
    for (size_t i=0;i<cols.size();++i) { 
        std::string c=cols[i]; 
        c.erase(0, c.find_first_not_of(" \t\n\r\f\v\'\"")); 
        c.erase(c.find_last_not_of(" \t\n\r\f\v\'\"")+1); 
        idx[c]= (int)i; 
    }
    std::string line;
    while (std::getline(f,line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields; 
        std::stringstream ss(line); 
        std::string fld;
        while (std::getline(ss, fld, ',')) fields.push_back(fld);
        ObsRecord r;
        auto get = [&](const std::string &name)->std::string { 
            if (idx.count(name) && idx[name] < (int)fields.size()) 
                return fields[idx[name]]; 
            return std::string(); 
        };
        
        std::string s_jd_iso = get("time_tdb_iso");
        std::string s_tt = get("time_tt_iso");
        std::string s_utc = get("time_utc_iso");
        
        // Вычисляем jd_tdb из time_utc_iso
        r.jd_tdb = NAN;
        if (!s_utc.empty()) {
            try {
                int Y,M,D,hh,mm; double ss;
                if (parse_iso_datetime(s_utc, Y,M,D,hh,mm,ss)) {
                    double utc1, utc2, tai1, tai2, tt1, tt2, tdb1, tdb2;
                    iauDtf2d("UTC", Y, M, D, hh, mm, ss, &utc1, &utc2);
                    iauUtctai(utc1, utc2, &tai1, &tai2);
                    iauTaitt(tai1, tai2, &tt1, &tt2);
                    
                    double dtr_hint = 0.0;
                    std::string sdtr = get("dtr_sec");
                    if (!sdtr.empty()) {
                        try { dtr_hint = std::stod(sdtr); } catch(...) {}
                    } else {
                        dtr_hint = compute_precise_dtr_iers2010(tt1 + tt2);
                    }
                    
                    iauTttdb(tt1, tt2, dtr_hint, &tdb1, &tdb2);
                    r.jd_tdb = tdb1 + tdb2;
                }
            } catch(...) {
                r.jd_tdb = NAN;
            }
        }
        
        std::string sra = get("ra_deg"); if (!sra.empty()) try { r.ra_deg = std::stod(sra); } catch(...) {}
        std::string sdec = get("dec_deg"); if (!sdec.empty()) try { r.dec_deg = std::stod(sdec); } catch(...) {}
        std::string sx = get("obs_x_km"); std::string sy = get("obs_y_km"); std::string sz = get("obs_z_km");
        if (!sx.empty()) try { r.obs_x_km = std::stod(sx); } catch(...) {}
        if (!sy.empty()) try { r.obs_y_km = std::stod(sy); } catch(...) {}
        if (!sz.empty()) try { r.obs_z_km = std::stod(sz); } catch(...) {}
        r.time_utc_iso = s_utc; r.time_tt_iso = s_tt; r.time_tdb_iso = s_jd_iso;
        std::string sut1 = get("ut1_utc"); if (!sut1.empty()) try{ r.ut1_utc = std::stod(sut1); } catch(...) { r.ut1_utc = 0.0; }
        std::string sxp = get("xp_arcsec"); if (!sxp.empty()) try{ r.xp_arcsec = std::stod(sxp); } catch(...) { r.xp_arcsec = 0.0; }
        std::string syp = get("yp_arcsec"); if (!syp.empty()) try{ r.yp_arcsec = std::stod(syp); } catch(...) { r.yp_arcsec = 0.0; }
        std::string sdtr = get("dtr_sec"); if (!sdtr.empty()) try { r.dtr_seconds = std::stod(sdtr); } catch(...) { r.dtr_seconds = NAN; }
        out.push_back(r);
    }
    return true;
}

static inline void itrs_to_gcrs(const Vector3d &obs_itrs_km, double tt1, double tt2, double ut11, double ut12, double xp_rad, double yp_rad, Vector3d &obs_gcrs_km) {
    double rc2t[3][3];
    iauC2t06a(tt1, tt2, ut11, ut12, xp_rad, yp_rad, rc2t);
    double rt[3][3];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) rt[i][j] = rc2t[j][i];
    double x = obs_itrs_km.x(), y = obs_itrs_km.y(), z = obs_itrs_km.z();
    obs_gcrs_km = Vector3d(rt[0][0]*x + rt[0][1]*y + rt[0][2]*z, rt[1][0]*x + rt[1][1]*y + rt[1][2]*z, rt[2][0]*x + rt[2][1]*y + rt[2][2]*z);
}

static inline void cartesian_to_spherical(const Vector3d &v_vec, double &ra_rad, double &dec_rad) {
    double p[3] = { v_vec.x(), v_vec.y(), v_vec.z() };
    double theta, phi;
    iauC2s(p, &theta, &phi);
    ra_rad = theta; dec_rad = phi;
}

static inline void apply_aberration(const Vector3d &v_obs_km_s, Vector3d &pnat_dir) {
    double pnat[3] = { pnat_dir.x(), pnat_dir.y(), pnat_dir.z() };
    double v[3] = { v_obs_km_s.x() / C_KM_S, v_obs_km_s.y() / C_KM_S, v_obs_km_s.z() / C_KM_S };
    double s = 1.0;
    double bm1 = 1.0;
    double ppr[3];
    iauAb(pnat, v, s, bm1, ppr);
    pnat_dir = Vector3d(ppr[0], ppr[1], ppr[2]);
}

static inline void apply_light_deflection(const Vector3d &pnat_dir_in, const Vector3d &sun_vec_from_obs_km, Vector3d &pnat_dir_out) {
    double p[3] = { pnat_dir_in.x(), pnat_dir_in.y(), pnat_dir_in.z() };
    double dist_km = sun_vec_from_obs_km.norm();
    if (dist_km < 1e-9) { pnat_dir_out = pnat_dir_in; return; }
    Eigen::Vector3d sun_to_obs = -sun_vec_from_obs_km;
    double edist = sun_to_obs.norm();
    double e[3] = { sun_to_obs.x()/edist, sun_to_obs.y()/edist, sun_to_obs.z()/edist };
    double em = edist / AU_KM;
    double pout[3];
    iauLdsun(p, e, em, pout);
    pnat_dir_out = Vector3d(pout[0], pout[1], pout[2]);
}

// -------------------- main --------------------
int main(int argc, char **argv) {
    std::string initfile = argv[1];
    std::string ephem_folder = argv[2];
    std::string obsfile = argv[3];
    double jd0; state_type X0;
    if (!read_initial_state(initfile, jd0, X0)) { 
        std::cerr << "Failed to read initial state from "<<initfile<<"\n"; 
        return 2; 
    }
    std::cout << "Initial JD0 = " << std::setprecision(12) << jd0 << "\n";
    std::vector<std::string> perturb = {"mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn"};
    IntegratorContext ctx;
    ctx.jd0 = jd0;
    ctx.gm = DEFAULT_GM;

    // --- ИНИЦИАЛИЗАЦИЯ РЕЛЯТИВИСТСКИХ ПАРАМЕТРОВ ---
    ctx.include_relativistic = true;
    ctx.include_lense_thirring = true;  // Для очень точных расчетов
    ctx.include_shapiro = true;         // Включать задержку Шапиро
    
    // Угловые моменты тел (для Лензе-Тирринга)
    ctx.body_angular_momentum["sun"] = Vector3d(0, 0, 1.92e41);
    ctx.body_angular_momentum["earth"] = Vector3d(0, 0, 7.05e33);
    ctx.body_angular_momentum["jupiter"] = Vector3d(0, 0, 6.9e38);

    for (const auto &b : perturb) {
        BodyEphemeris be(b);
        std::string path = ephem_folder + "/" + b + ".csv";
        if (!be.load_csv(path)) { 
            std::cerr << "Warning: failed to load ephemeris for "<<b<<" from "<<path<<" - skipping\n"; 
        } else { 
            ctx.ephems[b] = std::move(be); 
            std::cerr << "Loaded ephem for " << b << " with " << ctx.ephems[b].size() << " pts\n"; 
        }
    }
    
    BodyEphemeris sunEp; 
    std::string sunPath = ephem_folder + "/sun.csv";
    if (sunEp.load_csv(sunPath)) { 
        ctx.ephems["sun"] = std::move(sunEp); 
        ctx.ephems_are_barycentric = true; 
        std::cerr << "Loaded sun ephem \n"; 
    } else { 
        ctx.ephems_are_barycentric = false; 
        std::cerr << "No sun ephem found.\n"; 
    }

    // 1. Создание правой части уравнений (RHS)
    RHS rhs(ctx, jd0);
    
    // 2. Рекордер для записи состояния
    StepRecorder recorder;
    
    // 3. Создание адаптивного интегратора DOPRI5
    AdaptiveDOPRI5 integrator(1e-10, 1e-12, 1e-6, 1e4, 0.9, 5.0);
    
    // 4. Временные параметры
    double t0 = 0.0;
    double t1 = 3600.0 * 24.0 * 10.0; 
    double initial_step = 60.0; 
    
    // 5. Запуск интегрирования
    // Обертка для системы ОДУ
    auto system = [&](double t, const state_type& y, state_type& dydt) {
        rhs(t, y, dydt);
    };
    
    std::cerr << "Starting integration with DOPRI5 method...\n";
    integrator.integrate(system, X0, t0, t1, initial_step, recorder);
    std::cerr << "Integration complete. Steps recorded: " << recorder.times.size() << "\n";

    // построение сплайна
    SplineTraj traj; 
    traj.build_from_record(recorder.times, recorder.states, jd0);

    std::vector<ObsRecord> obs;
    if (!read_prepared_observations(obsfile, obs)) { 
        std::cerr << "Failed to read observations.\n"; 
        return 4; 
    }
    std::cerr << "Loaded " << obs.size() << " observations.\n";

    std::ofstream resf("residuals.csv");
    resf << "time_utc_iso,time_tdb_iso,jd_tdb,ra_obs_deg,dec_obs_deg,ra_mod_deg,dec_mod_deg,delta_ra_arcsec,delta_dec_arcsec\n";

    double sum_ra2 = 0.0, sum_dec2 = 0.0; 
    int nres = 0;

    // Обработка наблюдений
    for (auto &r : obs) {
        // 1. ВРЕМЯ: UTC -> TDB
        double jd_tdb = r.jd_tdb;
        double tt1, tt2, ut11, ut12;
        bool time_ok = false;
        double dtr_hint = std::isfinite(r.dtr_seconds) ? r.dtr_seconds : 0.0;
        if (!r.time_utc_iso.empty()) {
            time_ok = utciso_to_jdparts_and_tt_ut1(
                r.time_utc_iso, r.ut1_utc, dtr_hint, 
                jd_tdb, tt1, tt2, ut11, ut12);
        } 
        if (!time_ok) { 
            std::cerr << "Skipping obs with bad time: " << r.time_utc_iso << "\n"; 
            continue; 
        }

        // 2. КООРДИНАТЫ ОБСЕРВАТОРИИ (ITRF -> GCRS -> ICRS)
        if (!std::isfinite(r.obs_x_km) || !std::isfinite(r.obs_y_km) || !std::isfinite(r.obs_z_km)) {
            std::cerr << "Skipping obs with missing topocentric coordinates\n"; 
            continue;
        }
        // 2.1 Позиция обсерватории в ITRF (земная вращающаяся система)
        Vector3d obs_itrs(r.obs_x_km, r.obs_y_km, r.obs_z_km);
        // 2.2 Параметры вращения Земли (преобразование ITRF -> GCRS)
        double xp = r.xp_arcsec * (M_PI / 180.0 / 3600.0);  
        double yp = r.yp_arcsec * (M_PI / 180.0 / 3600.0);
        // 2.3 ITRF -> GCRS (инерциальная система, связанная с центром Земли)
        Vector3d obs_gcrs;
        itrs_to_gcrs(obs_itrs, tt1, tt2, ut11, ut12, xp, yp, obs_gcrs);
        // 2.4 GCRS -> ICRS (барицентрическая система)
        Vector3d earth_pos_icrs = Vector3d::Zero();
        Vector3d earth_vel_icrs = Vector3d::Zero();
        if (ctx.ephems.count("earth")) {
            earth_pos_icrs = ctx.ephems.at("earth").position_at(jd_tdb);
            auto vel_opt = ctx.ephems.at("earth").velocity_at(jd_tdb);
            if (vel_opt) {
                earth_vel_icrs = *vel_opt;
            }
        }
        double rnpb[3][3];
        iauPnm06a(tt1, tt2, rnpb);
        // Преобразуем obs_gcrs в ICRS
        Vector3d obs_gcrs_in_icrs(
            rnpb[0][0] * obs_gcrs.x() + rnpb[0][1] * obs_gcrs.y() + rnpb[0][2] * obs_gcrs.z(),
            rnpb[1][0] * obs_gcrs.x() + rnpb[1][1] * obs_gcrs.y() + rnpb[1][2] * obs_gcrs.z(),
            rnpb[2][0] * obs_gcrs.x() + rnpb[2][1] * obs_gcrs.y() + rnpb[2][2] * obs_gcrs.z()
        );
        // Положение обсерватории в ICRS
        Vector3d obs_pos_icrs = earth_pos_icrs + obs_gcrs_in_icrs;
        
        // 3. УРАВНЕНИЕ СВЕТОВОГО ВРЕМЕНИ
        std::pair<double, Vector3d> pr;
        if (ctx.include_shapiro) {
            // Для скорости обсерватории: скорость Земли + вращение
            const double OMEGA_E = 7.2921150e-5;  // рад/с
            Vector3d omega(0, 0, OMEGA_E);
            Vector3d rot_vel = omega.cross(obs_itrs);
            // Преобразуем скорость вращения в ICRS
            Vector3d rot_vel_icrs(
                rnpb[0][0] * rot_vel.x() + rnpb[0][1] * rot_vel.y() + rnpb[0][2] * rot_vel.z(),
                rnpb[1][0] * rot_vel.x() + rnpb[1][1] * rot_vel.y() + rnpb[1][2] * rot_vel.z(),
                rnpb[2][0] * rot_vel.x() + rnpb[2][1] * rot_vel.y() + rnpb[2][2] * rot_vel.z()
            );
            
            Vector3d obs_vel_icrs = earth_vel_icrs + rot_vel_icrs;
            pr = solve_light_time_relativistic(jd_tdb, obs_pos_icrs, traj, obs_vel_icrs, true);
        } 
        else {
            pr = solve_light_time(jd_tdb, obs_pos_icrs, traj);
        }
        double lt_sec = pr.first;
        Vector3d r_emit = pr.second;
        
        // 4. ВЫЧИСЛЕНИЕ НАПРАВЛЕНИЯ
        Vector3d obj_rel = r_emit - obs_pos_icrs;
        Vector3d pnat = obj_rel.normalized();
        
        // 5. РЕЛЯТИВИСТСКИЕ ПОПРАВКИ
        // 5.1 Отклонение света Солнцем
        Vector3d sun_pos_icrs = ctx.ephems.at("sun").position_at(jd_tdb);
        Vector3d sun_vec_obs = sun_pos_icrs - obs_pos_icrs;
        Vector3d pdef;
        apply_light_deflection(pnat, sun_vec_obs, pdef);
        // 5.2 Аберрация 
        Vector3d obs_vel_total = Vector3d::Zero();
        if (ctx.ephems.count("earth")) {
            auto ev_opt = ctx.ephems.at("earth").velocity_at(jd_tdb);
            if (ev_opt) {
                obs_vel_total = *ev_opt;
            }
        }
        // Добавляем вращение Земли
        const double OMEGA_E = 7.2921150e-5;
        Vector3d omega(0, 0, OMEGA_E);
        Vector3d rot_vel = omega.cross(obs_itrs);
        Vector3d rot_vel_icrs(
            rnpb[0][0] * rot_vel.x() + rnpb[0][1] * rot_vel.y() + rnpb[0][2] * rot_vel.z(),
            rnpb[1][0] * rot_vel.x() + rnpb[1][1] * rot_vel.y() + rnpb[1][2] * rot_vel.z(),
            rnpb[2][0] * rot_vel.x() + rnpb[2][1] * rot_vel.y() + rnpb[2][2] * rot_vel.z()
        );
        obs_vel_total += rot_vel_icrs;
        Vector3d ppr = pdef;
        apply_aberration(obs_vel_total, ppr);
        
        // 6. ПРЕОБРАЗОВАНИЕ В СФЕРИЧЕСКИЕ КООРДИНАТЫ
        double ra_rad, dec_rad;
        cartesian_to_spherical(ppr, ra_rad, dec_rad);
        double ra_mod_deg = ra_rad * 180.0 / M_PI;
        double dec_mod_deg = dec_rad * 180.0 / M_PI;
        
        // 7. СРАВНЕНИЕ С НАБЛЮДЕНИЯМИ
        if (!std::isfinite(r.ra_deg) || !std::isfinite(r.dec_deg)) {
            continue;
        }
        double dra_deg = r.ra_deg - ra_mod_deg;
        while (dra_deg > 180.0) dra_deg -= 360.0;
        while (dra_deg < -180.0) dra_deg += 360.0;
        double dra_arcsec = dra_deg * 3600.0 * std::cos(r.dec_deg * M_PI / 180.0);
        double ddec_arcsec = (r.dec_deg - dec_mod_deg) * 3600.0;
        
        // 8. ЗАПИСЬ РЕЗУЛЬТАТОВ
        resf << std::quoted(r.time_utc_iso) << "," << std::quoted(r.time_tdb_iso) << ","  << std::setprecision(12) << jd_tdb << "," << r.ra_deg << "," << r.dec_deg << "," << ra_mod_deg << "," << dec_mod_deg << "," << dra_arcsec << "," << ddec_arcsec << "\n";
        if (std::isfinite(dra_arcsec) && std::isfinite(ddec_arcsec)) {
            sum_ra2 += dra_arcsec * dra_arcsec;
            sum_dec2 += ddec_arcsec * ddec_arcsec;
            ++nres;
        }
    }
    resf.close();
    if (nres > 0) {
        double rms_ra = std::sqrt(sum_ra2 / nres);
        double rms_dec = std::sqrt(sum_dec2 / nres);
        std::cerr << "Residuals RMS: RA (arcsec) = " << rms_ra  << ", Dec (arcsec) = " << rms_dec << "\n";
    } 
    else {
        std::cerr << "No residuals to compute RMS.\n";
    }
    std::cerr << "Wrote residuals.csv\n";
    return 0;
}