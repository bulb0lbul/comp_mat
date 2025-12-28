

# python3 prepare_obs.py 1233.txt
from __future__ import annotations
import re
import math
import sys
import warnings
from typing import Dict, Tuple, Optional
import pandas as pd
from astropy.utils import iers
from astropy.coordinates import EarthLocation, GCRS
from astropy.time import Time
import astropy.units as u
import numpy as np

iers.conf.auto_download = False

EARTH_RADIUS_KM = 6371.0 

import warnings
try:
    import erfa
    warnings.filterwarnings("ignore", category=erfa.core.ErfaWarning)
except Exception:
    pass

def read_observatories_from_file() -> Dict[str, Tuple[float, float, float, str]]:
    """
    Читает файл obscodes.txt с переменной шириной колонок
    Формат: code longitude_hours cos sin place
    """
    obs_map = {}
    with open("obscodes.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Разбиваем строку по пробелам (любое количество)
            parts = line.split()
            if len(parts) < 5:
                continue
                
            code = parts[0]
            
            try:
                # longitude в часах, нужно перевести в градусы
                longitude_hours = float(parts[1])
                longitude_deg = longitude_hours * 15.0  # часов → градусов
                
                cos_val = float(parts[2])
                sin_val = float(parts[3])
                
                # Название обсерватории (может содержать пробелы)
                name = ' '.join(parts[4:])
                
                obs_map[code] = (
                    longitude_deg,  # В ГРАДУСАХ для obs2cart
                    cos_val,
                    sin_val,
                    name
                )
                
            except (ValueError, IndexError) as e:
                print(f"Ошибка парсинга строки: {line}")
                print(f"Ошибка: {e}")
                continue
    
    print(f"Загружено кодов обсерваторий: {len(obs_map)}")
    for code, (lon, cos, sin, name) in list(obs_map.items())[:5]:
        print(f"  {code}: lon={lon:.4f}°, cos={cos:.5f}, sin={sin:.5f}, name={name}")
    
    return obs_map

def obs2cart(obs_time: Time, lon_deg: float, cos_phi: float, sin_phi: float) -> np.ndarray:
    """
    Перевод координат обсерватории в GCRS.
    """
    # Вычисляем широту из sin/cos
    lat_rad = np.arctan2(sin_phi, cos_phi)
    
    # Создаём объект EarthLocation
    loc = EarthLocation(lat=lat_rad*u.rad, lon=lon_deg*u.deg, height=0*u.m)
    
    # Получаем координаты в GCRS на момент obs_time
    gcrs = loc.get_gcrs(obs_time)
    
    # Конвертируем из метров в километры
    x_km = gcrs.cartesian.x.value / 1000.0
    y_km = gcrs.cartesian.y.value / 1000.0
    z_km = gcrs.cartesian.z.value / 1000.0
    
    return np.array([x_km, y_km, z_km])


# ----------------- Регулярные выражения для парсинга -----------------
# дата: возможный префикс буквы, затем 4-цифр.год, месяц, день.фракция 
date_re = re.compile(r'([A-Za-z]?\d{4})\s+(\d{1,2})\s+(\d{1,2}\.\d+)', re.IGNORECASE)

# RA: hh mm ss.s  или компактный  hhmmss.s
ra_re = re.compile(r'(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)')
ra_compact_re = re.compile(r'\b(\d{6,7}(?:\.\d+)?)\b')

# Dec: +dd mm ss.s  или +dd mm  или компактный +ddmmss.s
dec_re = re.compile(r'([+\-]\d{1,2})\s+(\d{1,2}(?:\.\d+)?)\s+(\d{1,2}(?:\.\d+)?)')
dec_short_re = re.compile(r'([+\-]\d{1,2})\s+(\d{1,2}(?:\.\d+)?)')
dec_compact_re = re.compile(r'([+\-]\d{6,7}(?:\.\d+)?)')


def parse_time_from_line(line: str) -> Optional[Time]:
    """Извлекает дату/время из строки"""
    # Новое регулярное выражение для формата "A1927 10 03.92944"
    pattern = r'([A-Za-z]?\d{4})\s+(\d{1,2})\s+(\d{1,2}\.\d+)'
    m = re.search(pattern, line)
    
    if not m:
        return None
    
    year_str = m.group(1)
    # Удаляем букву префикса, если есть
    year_full = int(re.sub(r'[A-Za-z]', '', year_str))
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
    """Парсинг RA с учетом различных форматов."""
    # Формат 1: часы минуты секунды (разделены пробелами)
    pattern1 = r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)'
    # Формат 2: часы минуты (без секунд)
    pattern2 = r'(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)(?:\s+|$)'
    
    # Ищем после даты
    date_match = date_re.search(line)
    if not date_match:
        return None
    
    # Берем часть строки после даты
    date_end = date_match.end()
    rest = line[date_end:]
    
    # Пробуем первый формат
    m = re.search(pattern1, rest)
    if m:
        h = float(m.group(1))
        mn = float(m.group(2))
        sec = float(m.group(3)) if m.group(3) else 0.0
        return (h + mn/60.0 + sec/3600.0) * 15.0
    
    # Пробуем второй формат
    m = re.search(pattern2, rest)
    if m:
        h = float(m.group(1))
        mn = float(m.group(2))
        return (h + mn/60.0) * 15.0
    
    return None


def parse_dec_from_line(line: str) -> Optional[float]:
    """Парсинг Dec с учетом различных форматов."""
    # Формат 1: ±градусы минуты секунды
    pattern1 = r'([+\-]\d{1,2})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)'
    # Формат 2: ±градусы минуты
    pattern2 = r'([+\-]\d{1,2})\s+(\d{1,2}(?:\.\d+)?)(?:\s+|$)'
    
    date_match = date_re.search(line)
    if not date_match:
        return None
    
    date_end = date_match.end()
    rest = line[date_end:]
    
    # Сначала найдем RA, чтобы знать где начинается Dec
    ra_match = re.search(r'(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)(?:\s+(\d{1,2}(?:\.\d+)?))?', rest)
    if not ra_match:
        return None
    
    # Берем часть после RA
    ra_end = ra_match.end()
    dec_part = rest[ra_end:]
    
    # Ищем Dec
    m = re.search(pattern1, dec_part)
    if m:
        sgn = -1 if m.group(1).startswith('-') else 1
        deg = abs(float(re.sub(r'[+\-]', '', m.group(1))))
        arcmin = float(m.group(2))
        arcsec = float(m.group(3)) if m.group(3) else 0.0
        return sgn * (deg + arcmin/60.0 + arcsec/3600.0)
    
    m = re.search(pattern2, dec_part)
    if m:
        sgn = -1 if m.group(1).startswith('-') else 1
        deg = abs(float(re.sub(r'[+\-]', '', m.group(1))))
        arcmin = float(m.group(2))
        return sgn * (deg + arcmin/60.0)
    
    return None

def detect_obs_code_in_line(line: str, obs_map: Dict[str, Tuple[float, float, float, str]]) -> Optional[str]:
    """ Ищет код обсерватории."""
    if not line:
        return None

    # Удаляем пробелы в начале и конце
    trimmed = line.strip()
    
    # Разбиваем строку на части по пробелам
    parts = trimmed.split()
    
    if not parts:
        return None
    
    # Последняя часть - это ссылка (например, "HD016024")
    last_part = parts[-1]
    
    # Ищем 3 цифры в конце последней части
    # Это может быть код обсерватории
    match = re.search(r'(\d{3})$', last_part)
    if match:
        code = match.group(1)
        if code in obs_map:
            return code
    
    # Если не нашли 3 цифры, пробуем весь последний токен
    if last_part in obs_map:
        return last_part
    
    # Или предпоследний токен (на случай, если код отдельно)
    if len(parts) >= 2:
        prev_part = parts[-2]
        if prev_part in obs_map:
            return prev_part
    
    return None


def find_obs_code_fallback_from_token(token: str, obs_map: Dict[str, Tuple[float, float, float, str]]) -> Optional[str]:
    """ Ищет код обсерватории как последние 3 символа предоставленного токена."""
    if not token:
        return None

    tok = token.strip()
    if len(tok) < 3:
        return None

    last_three = tok[-3:]
    if last_three in obs_map:
        return last_three

    return None


def parse_observations_file_improved(path: str, obs_map: Dict[str, Tuple[float, float, float, str]]):
    """
    Основной парсер - ищет код обсерватории как последние 3 символа строки.
    """
    records = []
    bad_lines = []
    counters = {"lines_total": 0, "date_found": 0, "ra_found": 0, "dec_found": 0, "obs_found": 0}

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
            if t_utc is not None:
                counters["date_found"] += 1

            ra_deg = parse_ra_from_line(L)
            if ra_deg is not None:
                counters["ra_found"] += 1

            dec_deg = parse_dec_from_line(L)
            if dec_deg is not None:
                counters["dec_found"] += 1

            if t_utc is None:
                bad_lines.append(L.strip() + "  <-- date not found")
                continue

            if ra_deg is None or dec_deg is None:
                bad_lines.append(L.strip() + "  <-- RA/Dec not found")
                continue

            # Ищем код обсерватории как последние 3 символа строки
            obs_code = detect_obs_code_in_line(L, obs_map=obs_map)
            
            if obs_code is None:
                # Фолбэк: ищем в последнем токене
                last_tok = re.split(r'\s+', L.strip())[-1]
                obs_code = find_obs_code_fallback_from_token(last_tok, obs_map)
                
            if obs_code is not None:
                counters["obs_found"] += 1

            if obs_code is None:
                bad_lines.append(L.strip() + "  <-- obs code NOT found")
                continue

            lon_deg, cos_phi, sin_phi, obs_name = obs_map[obs_code]
            obs_pos = obs2cart(t_utc, lon_deg, cos_phi, sin_phi)
            obs_x, obs_y, obs_z = obs_pos[0], obs_pos[1], obs_pos[2]
            t_tdb = t_utc.tdb
            
            rec = {
                "time_utc": t_utc,
                "time_tdb": t_tdb,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "obs_x": obs_x,
                "obs_y": obs_y,
                "obs_z": obs_z,
                "obs_code": obs_code,
                "obs_name": obs_name
            }
            records.append(rec)

        except Exception as e:
            bad_lines.append(f"{L.strip()}  <-- PARSE ERROR: {e}")
            continue

    df = pd.DataFrame(records)
    print("=== PARSER DIAGNOSTICS ===")
    for k, v in counters.items():
        print(f"{k}: {v}")
    print(f"Успешно распознано записей: {len(df)}")
    print(f"Ошибочных строк: {len(bad_lines)}")
    print("==========================")
    return df, bad_lines


def convert_date_format(date_str: str) -> str:
    """
    Конвертирует дату из формата DD-MM-YYYY в YYYY-MM-DD
    """
    try:
        day, month, year = date_str.split('-')
        return f"{year}-{month}-{day}"
    except Exception as e:
        print(f"Ошибка при конвертации даты '{date_str}': {e}")
        raise ValueError(f"Неверный формат даты: {date_str}. Ожидается DD-MM-YYYY")

def filter_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        # Конвертируем даты из DD-MM-YYYY в YYYY-MM-DD
        start_date_iso = convert_date_format(start_date)
        end_date_iso = convert_date_format(end_date)
        
        # Преобразуем строки в объекты Time в UTC
        start_time_utc = Time(f"{start_date_iso}T00:00:00", format='isot', scale='utc')
        end_time_utc = Time(f"{end_date_iso}T23:59:59.999", format='isot', scale='utc')
        
        print(f"Фильтрация данных по периоду в UTC: {start_date} - {end_date}")
        print(f"Начальное время (UTC): {start_time_utc.iso}")
        print(f"Конечное время (UTC): {end_time_utc.iso}")
        
        # Фильтруем данные по UTC времени
        mask = (df['time_utc'] >= start_time_utc) & (df['time_utc'] <= end_time_utc)
        filtered_df = df[mask].copy()
        
        print(f"Записей до фильтрации: {len(df)}")
        print(f"Записей после фильтрации: {len(filtered_df)}")
        
        if filtered_df.empty:
            print("ВНИМАНИЕ: После фильтрации не осталось данных!")
        
        return filtered_df
    
    except Exception as e:
        print(f"Ошибка при фильтрации по дате: {e}")
        return df


def write_output_table(df: pd.DataFrame, output_file: str = "output.txt"):
    """
    Формат:
    utc_jd tdb_jd ra_deg dec_deg obs_x obs_y obs_z
    """
    if df.empty:
        print("Нет данных для записи")
        return

    df = df.sort_values("time_utc")

    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            utc_jd = row["time_utc"].jd
            tdb_jd = row["time_tdb"].jd

            f.write(
                f"{utc_jd:.10f} "
                f"{tdb_jd:.10f} "
                f"{row['ra_deg']:.10f} "
                f"{row['dec_deg']:.10f} "
                f"{row['obs_x']:.10f} "
                f"{row['obs_y']:.10f} "
                f"{row['obs_z']:.10f}\n"
            )

    print(f"Данные успешно сохранены в {output_file}")
    print(f"Всего записей в файле: {len(df)}")



def main(infile: str, outfile: str = "output.txt", badfile: str = "bad_lines.txt"):
    print("Чтение таблицы observatory codes из файла obscodes.txt...")
    try:
        obs_map = read_observatories_from_file()
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        sys.exit(1)

    print("Парсинг входного файла...")
    df, bad = parse_observations_file_improved(infile, obs_map)
    
    if not df.empty:
        df_filtered = filter_by_date_range(df, "30-09-2022", "7-10-2022")
        
        if not df_filtered.empty:
            write_output_table(df_filtered, outfile)
        else:
            print("Нет данных в указанном временном периоде для записи в файл")
    
    if bad:
        with open(badfile, 'w', encoding='utf-8') as f:
            for L in bad:
                f.write(L.rstrip() + "\n")
        print(f"Ошибки сохранены в {badfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python prepare_obs.py <input_file> [output_file]")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) >= 3 else "output.txt"
    main(infile, outfile)
