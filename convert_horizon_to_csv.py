# python3 convert_horizon_to_csv.py sun.txt

from __future__ import annotations
import sys
import re
import numpy as np
import pandas as pd
from math import radians, cos, sin

def parse_horizons_file(path: str):
    """Parse Horizons vector-table file between $$SOE and $$EOE"""
    jd = []
    iso = []
    xs = []; ys = []; zs = []
    vxs = []; vys = []; vzs = []
    in_table = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('$$SOE'):
                in_table = True
                continue
            if line.startswith('$$EOE'):
                in_table = False
                break
            if not in_table:
                continue
            parts = [p.strip() for p in line.split(',') if p.strip() != '']
            if len(parts) < 8:
                tokens = re.split(r'\s+', line)
                nums = []
                for tok in tokens:
                    if re.match(r'^[+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?$', tok):
                        nums.append(tok)
                if len(nums) >= 7:
                    try:
                        jd_val = float(nums[0])
                        date_str = ""
                        x = float(nums[1]); y = float(nums[2]); z = float(nums[3])
                        vx = float(nums[4]); vy = float(nums[5]); vz = float(nums[6])
                    except Exception:
                        continue
                else:
                    continue
            else:
                try:
                    jd_val = float(parts[0])
                except Exception:
                    continue
                date_str = parts[1]
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    vx = float(parts[5])
                    vy = float(parts[6])
                    vz = float(parts[7])
                except Exception:
                    def toflt(s):
                        s2 = re.sub(r'[^0-9eE\+\-\.]', '', s)
                        return float(s2)
                    try:
                        x = toflt(parts[2]); y = toflt(parts[3]); z = toflt(parts[4])
                        vx = toflt(parts[5]); vy = toflt(parts[6]); vz = toflt(parts[7])
                    except Exception:
                        continue

            jd.append(jd_val)
            iso.append(date_str)
            xs.append(x); ys.append(y); zs.append(z)
            vxs.append(vx); vys.append(vy); vzs.append(vz)

    df = pd.DataFrame({
        'jd_tdb': jd,
        'date_tdb': iso,
        'x_km': xs,
        'y_km': ys,
        'z_km': zs,
        'vx_km_s': vxs,
        'vy_km_s': vys,
        'vz_km_s': vzs
    })
    return df

def eclipticJ2000_to_icrs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Правильное преобразование из эклиптики J2000 в ICRS (J2000).
    Используем вращение вокруг оси Y на угол epsilon (наклон эклиптики).
    Формула: R_y(epsilon) где epsilon = 23.43929111°
    """
    epsilon_deg = 23.43929111
    eps = radians(epsilon_deg)
    ce = cos(eps)
    se = sin(eps)

    x = df['x_km'].to_numpy()
    y = df['y_km'].to_numpy()
    z = df['z_km'].to_numpy()
    vx = df['vx_km_s'].to_numpy()
    vy = df['vy_km_s'].to_numpy()
    vz = df['vz_km_s'].to_numpy()

    # Вращение вокруг оси Y: R_y(epsilon)
    # Матрица: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    x_i =  ce * x + se * z
    y_i =  y
    z_i = -se * x + ce * z
    
    vx_i =  ce * vx + se * vz
    vy_i =  vy
    vz_i = -se * vx + ce * vz

    # Создаем DataFrame с преобразованными координатами
    out = pd.DataFrame({
        'JDTDB': df['jd_tdb'],
        'X': x_i,
        'Y': y_i,
        'Z': z_i,
        'VX': vx_i,
        'VY': vy_i,
        'VZ': vz_i
    })
    
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_horizon_to_csv.py <horizons_file.txt> [output_prefix]")
        sys.exit(1)
    infile = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) >= 3 else infile.rsplit('.', 1)[0]

    print("Парсинг файла Horizons:", infile)
    df = parse_horizons_file(infile)
    if df.empty:
        print("Не удалось распарсить данные. Проверьте формат файла.")
        sys.exit(2)

    print(f"Распаршено {len(df)} строк")
    df_icrs = eclipticJ2000_to_icrs(df)
    
    # Формат: JDTDB, X, Y, Z, VX, VY, VZ
    output_csv = f"ephem_folder/{prefix}.csv"
    df_icrs.to_csv(output_csv, index=False, float_format='%.12e')
    
    print(f"\nСохранено в: {output_csv}")

if __name__ == '__main__':
    main()