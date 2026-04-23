"""
LOTSA 큐레이션 다운로드: 도메인 다양성 우선 선택

기존 다운로드 + climate 정리 후, 미수신 중요 데이터셋만 선별 추가.
"""
import os, time
from huggingface_hub import HfApi, hf_hub_download

SAVE_DIR = '/workspace/HypOPFN2/dataset/lotsa'

# 도메인별 선별된 추가 다운로드 목록
CURATED_ADDITIONAL = [
    # load_forecasting (전력 부하 예측)
    'gfc14_load', 'hog', 'largest_2021', 'spain',
    # transport (교통)
    'taxi_30min', 'uber_tlc_hourly', 'traffic_hourly',
    # economic (경제)
    'favorita_sales', 'fred_md', 'hierarchical_sales',
    # energy_buildings (건물 에너지)
    'london_smart_meters_with_missing', 'kdd_cup_2018_with_missing', 'residential_pv_power',
    # energy_env (환경 에너지)
    'solar_power', 'wind_power',
    # health_misc (보건/기타)
    'hospital', 'nn5_daily_with_missing', 'restaurant',
    # misc_tech
    'tourism_monthly',
    # other (다양한 도메인)
    'm5', 'pedestrian_counts', 'project_tycho', 'subseasonal',
    'sunspot_with_missing', 'temperature_rain_with_missing', 'us_births',
    # web
    'godaddy', 'wiki-rolling_nips',
]

api = HfApi()
downloaded = set(os.listdir(SAVE_DIR))
to_download = [d for d in CURATED_ADDITIONAL if d not in downloaded]

print(f'Curated list: {len(CURATED_ADDITIONAL)}')
print(f'Already have: {len(CURATED_ADDITIONAL) - len(to_download)}')
print(f'To download: {len(to_download)}')
print()

t0 = time.time()
done_count, errors = 0, 0

for i, ds_name in enumerate(to_download):
    try:
        files = list(api.list_repo_tree(
            'Salesforce/lotsa_data',
            repo_type='dataset',
            path_in_repo=ds_name,
            recursive=True
        ))
        arrow_files = [f for f in files if hasattr(f, 'path') and f.path.endswith('.arrow')]

        if not arrow_files:
            print(f'  [{i+1}/{len(to_download)}] {ds_name} ⚠ (no arrow files)')
            continue

        ds_dir = os.path.join(SAVE_DIR, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        for af in arrow_files:
            local_path = os.path.join(SAVE_DIR, af.path)
            if os.path.exists(local_path):
                continue
            hf_hub_download(
                repo_id='Salesforce/lotsa_data',
                filename=af.path,
                repo_type='dataset',
                local_dir=SAVE_DIR,
            )

        done_count += 1
        elapsed = time.time() - t0
        size_mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(ds_dir)
            for f in fs
        ) / 1e6
        print(f'  [{i+1}/{len(to_download)}] {ds_name} ✓ ({size_mb:.0f}MB, {elapsed:.0f}s)')
    except Exception as e:
        errors += 1
        print(f'  [{i+1}/{len(to_download)}] {ds_name} ✗ ({type(e).__name__}: {str(e)[:100]})')

print(f'\nDone: {done_count} downloaded, {errors} errors, {time.time()-t0:.0f}s')

# Total size
total = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(SAVE_DIR)
    for f in fs
)
print(f'LOTSA total size: {total/1e9:.1f}GB')
print(f'Dataset count: {len([d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))])}')
