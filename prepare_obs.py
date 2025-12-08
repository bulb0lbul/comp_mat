# python3 prepare_obs.py 1233.txt
from __future__ import annotations
import re
import math
import sys
import warnings
from typing import Dict, Tuple, Optional

import requests
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False

EARTH_EQUATORIAL_RADIUS_KM = 6378.140

# URL для ObsCodesF
MPC_OBSCODES_URLS = [
    "https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html",
    "https://www.projectpluto.com/neocp2/ObsCodesF.html"
]


import warnings
try:
    import erfa
    warnings.filterwarnings("ignore", category=erfa.core.ErfaWarning)
except Exception:
    pass


def fetch_obs_codes_html() -> str:
    """ Скачать HTML-страницу с ObsCodesF """
    last_err = None
    for url in MPC_OBSCODES_URLS:
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Не удалось загрузить список ObsCodesF. Ошибка: {last_err}")


def parse_obs_codes_from_html(html_text: str) -> Dict[str, Tuple[float, float, float, str]]:
    """
    Парсит HTML-страницу ObsCodesF и возвращает словарь:
        code -> (longitude_deg, rho_cos, rho_sin, name)
    rho_cos и rho_sin в единицах земных экваториальных радиусов (как на странице MPC).
    """
    lines = html_text.splitlines()
    out: Dict[str, Tuple[float, float, float, str]] = {}
    # Регекс для строк данных: код, longitude, cos, sin, name
    data_re = re.compile(r'^\s*([0-9A-Z]{3})\s+([\-0-9\.]+)\s+([\-0-9\.]+)\s+([+\-]?[0-9\.]+)\s+(.*)$')
    for L in lines:
        m = data_re.match(L)
        if m:
            code = m.group(1).strip().upper()
            try:
                lon = float(m.group(2))
                rho_cos = float(m.group(3))
                rho_sin = float(m.group(4))
            except Exception:
                continue
            name = m.group(5).strip()
            out[code] = (lon, rho_cos, rho_sin, name)
    if not out:
        raise RuntimeError("Не найдено данных обсерваторий в загруженном HTML.")
    return out


# ----------------- Регулярные выражения для парсинга -----------------
# дата: возможный префикс буквы, затем 4-цифр.год, месяц, день.фракция 
date_re = re.compile(r'[A-Za-z]?((?:18|19|20)\d{2})\s+(\d{1,2})\s+(\d{1,2}\.\d+)', re.IGNORECASE)

# RA: hh mm ss.s  или компактный  hhmmss.s
ra_re = re.compile(r'(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)')
ra_compact_re = re.compile(r'\b(\d{6,7}(?:\.\d+)?)\b')

# Dec: +dd mm ss.s  или +dd mm  или компактный +ddmmss.s
dec_re = re.compile(r'([+\-]\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)')
dec_short_re = re.compile(r'([+\-]\d{1,2})\s+(\d{1,2}(?:\.\d+)?)')
dec_compact_re = re.compile(r'([+\-]\d{6,7}(?:\.\d+)?)')

# alt-vector: три подряд signed числа, допускаем пробел после знака: "+ 3440.2430  + 1889.8250  + 4667.1340"
vector_re = re.compile(r'([+\-]\s*\d{1,7}(?:\.\d+)?)\s+([+\-]\s*\d{1,7}(?:\.\d+)?)\s+([+\-]\s*\d{1,7}(?:\.\d+)?)')


def parse_time_from_line(line: str) -> Optional[Time]:
    """Извлекает дату/время (UTC) из строки; возвращает astropy.Time или None."""
    m = date_re.search(line)
    if not m:
        return None
    year_full = int(m.group(1))
    month = int(m.group(2))
    dayf = float(m.group(3))
    day_int = int(math.floor(dayf))
    frac = dayf - day_int
    seconds = frac * 86400.0
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = (seconds % 60)
    iso = f"{year_full:04d}-{month:02d}-{day_int:02d}T{hh:02d}:{mm:02d}:{ss:09.6f}"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            t = Time(iso, format='isot', scale='utc')
        return t
    except Exception:
        return None


def parse_ra_from_line(line: str) -> Optional[float]:
    """Ищет RA; возвращает градусы (J2000) или None."""
    m = ra_re.search(line)
    if m:
        h = float(m.group(1)); mn = float(m.group(2)); sec = float(m.group(3))
        return (h + mn / 60.0 + sec / 3600.0) * 15.0
    m2 = ra_compact_re.search(line)
    if m2:
        s = m2.group(1)
        part = s.split('.')[0]
        if len(part) >= 6:
            hh = float(part[0:2]); mm = float(part[2:4]); ss = float(part[4:]) if len(part) > 4 else 0.0
            return (hh + mm / 60.0 + ss / 3600.0) * 15.0
    return None


def parse_dec_from_line(line: str) -> Optional[float]:
    """Ищет Declination; возвращает градусы (signed) или None."""
    m = dec_re.search(line)
    if m:
        sgn = -1 if m.group(1).startswith('-') else 1
        deg_abs = abs(float(re.sub(r'[+\-]', '', m.group(1))))
        arcmin = float(m.group(2)); arcsec = float(m.group(3))
        return sgn * (deg_abs + arcmin / 60.0 + arcsec / 3600.0)
    m2 = dec_short_re.search(line)
    if m2:
        sgn = -1 if m2.group(1).startswith('-') else 1
        deg_abs = abs(float(re.sub(r'[+\-]', '', m2.group(1))))
        arcmin = float(m2.group(2))
        return sgn * (deg_abs + arcmin / 60.0)
    m3 = dec_compact_re.search(line)
    if m3:
        s = m3.group(1); sgn = -1 if s.startswith('-') else 1; s2 = s[1:]
        if len(s2) >= 6:
            d = float(s2[0:2]); m_ = float(s2[2:4]); sec = float(s2[4:]) if len(s2) > 4 else 0.0
            return sgn * (d + m_ / 60.0 + sec / 3600.0)
    return None


def detect_obs_code_in_line(line: str, obs_map: Dict[str, Tuple[float, float, float, str]]) -> Optional[str]:
    """
    Попытка найти код обсерватории как отдельный токен (из списка obs_map).
    Возвращает код в верхнем регистре или None.
    """
    tokens = re.split(r'\s+', line.strip())
    for tok in reversed(tokens):
        clean = re.sub(r'[^0-9A-Z]', '', tok.upper())
        if clean in obs_map:
            return clean
    # fallback: искать подстроки
    for code in obs_map.keys():
        if code in line:
            return code
    return None


def find_obs_code_fallback_from_token(token: str, obs_map: Dict[str, Tuple[float, float, float, str]]) -> Optional[str]:
    """Пытаемся извлечь 3-символьную подстроку из токена, совпадающую с ключом obs_map."""
    clean = re.sub(r'[^0-9A-Z]', '', token.upper())
    n = len(clean)
    for i in range(max(0, n - 3 + 1)):
        sub = clean[i:i + 3]
        if len(sub) == 3 and sub in obs_map:
            return sub
    if n >= 3:
        tail = clean[-3:]
        if tail in obs_map:
            return tail
    return None


def cylindrical_to_cartesian_km(lon_deg: float, r_radii: float, z_radii: float) -> Tuple[float, float, float]:
    """
    λ (deg), r (земные радиусы), z (земные радиусы) -> X,Y,Z в км.
    Преобразует цилиндрические координаты обсерватории в декартовы
    """
    lon_rad = math.radians(lon_deg)
    r_km = r_radii * EARTH_EQUATORIAL_RADIUS_KM
    z_km = z_radii * EARTH_EQUATORIAL_RADIUS_KM
    x = r_km * math.cos(lon_rad)
    y = r_km * math.sin(lon_rad)
    z = z_km
    return x, y, z


def compute_dtr_ephemeris(t_tt):
    """
    Вычисление точной разницы TT-TDB (в секундах) через эфемериды.
    
    Использует астрономические модели для вычисления dtr = TT - TDB.
    Более точная альтернатива аналитической модели.
    """
    # Создаем время в шкале TT
    tt_time = Time(t_tt.jd, format='jd', scale='tt')
        
    # Преобразуем в TDB 
    tdb_time = tt_time.tdb
        
    # Разница в секундах: dtr = TT - TDB
    dtr_seconds = (tt_time.jd - tdb_time.jd) * 86400.0
        
    return dtr_seconds


def parse_observations_file_improved(path: str, obs_map: Dict[str, Tuple[float, float, float, str]]):
    """
    Основной парсер:
    - находит date/RA/Dec или alt-vector
    - пытается распознать obs_code
    - сохраняет флаги и возвращает DataFrame + список bad_lines
    """
    records = []
    bad_lines = []
    counters = {"lines_total": 0, "date_found": 0, "ra_found": 0, "dec_found": 0, "alt_vector": 0, "obs_found": 0}

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    start = 0
    for i, L in enumerate(lines[:12]):
        if re.search(r'Date\s*\(UT\)', L, re.IGNORECASE) or re.search(r'J2000 RA', L, re.IGNORECASE):
            start = i + 1
            break

    for L in lines[start:]:
        if not L.strip():
            continue
        counters["lines_total"] += 1
        try:
            t_utc = parse_time_from_line(L)
            # Ищет паттерн: "2023 10 15.92734"
            # преобразует в "2023-10-15T22:15:22.333"
            if t_utc is not None:
                counters["date_found"] += 1

            ra_deg = parse_ra_from_line(L)
            # "123456.7" превращает в 12ч 34м 56.7с
            # Конвертация: (12 + 34/60 + 56.7/3600) × 15 = 188.73625°
            if ra_deg is not None:
                counters["ra_found"] += 1

            dec_deg = parse_dec_from_line(L)
            # "+451234.5" в  +45° 12' 34.5"
            # Конвертация: 45 + 12/60 + 34.5/3600 = 45.209583°
            if dec_deg is not None:
                counters["dec_found"] += 1

            alt_x = alt_y = alt_z = None
            mvec = vector_re.search(L)
            # Иногда вместо RA/Dec могут быть геоцентрические координаты
            if mvec:
                try:
                    # + 3440.2430  + 1889.8250  + 4667.1340
                    def norm_num(s): return float(s.replace(' ', ''))
                    alt_x = norm_num(mvec.group(1)); alt_y = norm_num(mvec.group(2)); alt_z = norm_num(mvec.group(3))
                    counters["alt_vector"] += 1
                except Exception:
                    alt_x = alt_y = alt_z = None

            # если нет даты — ошибкв
            if t_utc is None:
                bad_lines.append(L.strip() + "  <-- date not found")
                continue

            # если нет ни RA/Dec, ни alt-vector — ошибка
            if ra_deg is None and dec_deg is None and alt_x is None:
                bad_lines.append(L.strip() + "  <-- RA/Dec and alt-vector not found")
                continue

            # попытка найти obs_code
            obs_code = detect_obs_code_in_line(L, obs_map=obs_map)
            if obs_code is None:
                last_tok = re.split(r'\s+', L.strip())[-1]
                obs_code = find_obs_code_fallback_from_token(last_tok, obs_map)
            if obs_code is not None:
                counters["obs_found"] += 1

            if obs_code is None:
                bad_lines.append(L.strip() + "  <-- obs code NOT found; record saved with obs_code=None")

            if obs_code is not None:
                lon_deg, r_radii, z_radii, obs_name = obs_map[obs_code]
                obs_x_km, obs_y_km, obs_z_km = cylindrical_to_cartesian_km(lon_deg, r_radii, z_radii)
            else:
                lon_deg = r_radii = z_radii = None
                obs_name = None
                obs_x_km = obs_y_km = obs_z_km = None

            t_tt = t_utc.tt
            t_tdb = t_utc.tdb
            
            dtr_sec = compute_dtr_ephemeris(t_tt)
            
            rec = {
                "time_utc_iso": t_utc.iso,
                "time_tt_iso": t_tt.iso,
                "time_tdb_iso": t_tdb.iso,
                "dtr_sec": dtr_sec, 
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "alt_x_raw": alt_x,
                "alt_y_raw": alt_y,
                "alt_z_raw": alt_z,
                "obs_code": obs_code,
                "obs_name": obs_name,
                "obs_x_km": obs_x_km,
                "obs_y_km": obs_y_km,
                "obs_z_km": obs_z_km
            }
            records.append(rec)

        except Exception as e:
            bad_lines.append(f"{L.strip()}  <-- PARSE ERROR: {e}")
            continue

    df = pd.DataFrame(records)
    print("=== PARSER DIAGNOSTICS ===")
    for k, v in counters.items():
        print(f"{k}: {v}")
    print("==========================")
    return df, bad_lines


def main(infile: str, outfile_csv: str = "all_observations.csv", badfile: str = "bad_lines.txt"):
    print("Загрузка таблицы observatory codes (ObsCodesF) ...")
    html = fetch_obs_codes_html()
    obs_map = parse_obs_codes_from_html(html)
    print(f"Загружено кодов обсерваторий: {len(obs_map)}")

    print("Парсинг входного файла...")
    df, bad = parse_observations_file_improved(infile, obs_map)
    print(f"Успешно распознано записей: {len(df)}. Ошибочных строк: {len(bad)}")

    df.to_csv(outfile_csv, index=False)
    with open(badfile, 'w', encoding='utf-8') as f:
        for L in bad:
            f.write(L.rstrip() + "\n")
    print(f"Сохранено: {outfile_csv}; Ошибки в {badfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python prepare_obs.py <input_file> [output_csv]")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) >= 3 else "all_observations.csv"
    main(infile, outfile)
