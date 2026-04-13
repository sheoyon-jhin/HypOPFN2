"""
LOTSA 다운로드 (eval benchmark 제외)

제외: weather, oikolab_weather, m1_*, m3_*, m4_*
나머지 ~159 datasets 다운로드

Usage: python data/download_lotsa.py
"""
import os
from huggingface_hub import HfApi, hf_hub_download

SAVE_DIR = '/workspace/HypOPFN/dataset/lotsa'
os.makedirs(SAVE_DIR, exist_ok=True)

# Eval benchmarks to EXCLUDE
EXCLUDE = {
    'weather', 'oikolab_weather',
    'm1_monthly', 'm1_quarterly', 'm1_yearly',
    'm3_monthly', 'm3_quarterly', 'm3_yearly', 'monash_m3_monthly',
    'monash_m3_other', 'monash_m3_quarterly', 'monash_m3_yearly',
    'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_weekly', 'm4_yearly',
}

api = HfApi()
all_items = list(api.list_repo_tree('Salesforce/lotsa_data', repo_type='dataset', recursive=False))
datasets = [item.path for item in all_items if not item.path.startswith('.')]

print(f'Total LOTSA datasets: {len(datasets)}')
included = [d for d in datasets if d not in EXCLUDE]
excluded = [d for d in datasets if d in EXCLUDE]
print(f'Excluded (eval benchmarks): {len(excluded)} — {excluded}')
print(f'Included: {len(included)}')

# Download each dataset's arrow files
import time
t0 = time.time()
downloaded = 0
errors = 0

for i, ds_name in enumerate(included):
    try:
        # List files in this dataset folder
        files = list(api.list_repo_tree(
            'Salesforce/lotsa_data',
            repo_type='dataset',
            path_in_repo=ds_name,
            recursive=True
        ))
        arrow_files = [f for f in files if hasattr(f, 'path') and f.path.endswith('.arrow')]

        ds_dir = os.path.join(SAVE_DIR, ds_name)
        os.makedirs(ds_dir, exist_ok=True)

        for af in arrow_files:
            local_path = os.path.join(SAVE_DIR, af.path)
            if os.path.exists(local_path):
                continue  # skip if already downloaded
            hf_hub_download(
                repo_id='Salesforce/lotsa_data',
                filename=af.path,
                repo_type='dataset',
                local_dir=SAVE_DIR,
            )
        downloaded += 1
        elapsed = time.time() - t0
        if (i+1) % 10 == 0 or i < 5:
            print(f'  [{i+1}/{len(included)}] {ds_name} ✓ ({elapsed:.0f}s)')
    except Exception as e:
        errors += 1
        print(f'  [{i+1}/{len(included)}] {ds_name} ✗ ({e})')

print(f'\nDone: {downloaded} downloaded, {errors} errors, {time.time()-t0:.0f}s')
print(f'Saved to: {SAVE_DIR}')

# Check total size
total_size = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))
print(f'Total size: {total_size/1e9:.1f} GB')
