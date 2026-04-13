#!/opt/miniforge3/bin/python3.13
"""
Generate synthetic data using TempoPFN's generators natively (same as their pipeline).
Univariate output, same proportions as TempoPFN configs/example.yaml.

Bypasses audio generators (pyo) and cauker (cupy) that have missing dependencies.
Uses TempoPFN's GeneratorWrapper pattern directly.

Usage:
  python gen_tempopfn_native.py                           # All generators, 3 base batches
  python gen_tempopfn_native.py --num-batches 5           # More data
  python gen_tempopfn_native.py --types ou_process step   # Specific generators
  python gen_tempopfn_native.py --length 1024             # Our length
"""
import sys, os, gc, argparse, time, logging
sys.path.insert(0, '/workspace/TempoPFN')

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path

# Import generators directly (avoid audio/cauker imports)
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import ForecastPFNGeneratorWrapper
from src.synthetic_generation.gp_prior.gp_generator_wrapper import GPGeneratorWrapper
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import KernelGeneratorWrapper
from src.synthetic_generation.sawtooth.sawtooth_generator_wrapper import SawToothGeneratorWrapper
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import SineWaveGeneratorWrapper
from src.synthetic_generation.steps.step_generator_wrapper import StepGeneratorWrapper
from src.synthetic_generation.spikes.spikes_generator_wrapper import SpikesGeneratorWrapper
from src.synthetic_generation.anomalies.anomaly_generator_wrapper import AnomalyGeneratorWrapper
from src.synthetic_generation.ornstein_uhlenbeck_process.ou_generator_wrapper import OrnsteinUhlenbeckProcessGeneratorWrapper

from src.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    SawToothGeneratorParams,
    SineWaveGeneratorParams,
    StepGeneratorParams,
    SpikesGeneratorParams,
    AnomalyGeneratorParams,
    OrnsteinUhlenbeckProcessGeneratorParams,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TempoPFN original proportions
GENERATOR_CONFIGS = {
    'forecast_pfn': (ForecastPFNGeneratorParams, ForecastPFNGeneratorWrapper, 1.0),
    'gp':           (GPGeneratorParams, GPGeneratorWrapper, 1.0),
    'kernel':       (KernelGeneratorParams, KernelGeneratorWrapper, 1.0),
    'sawtooth':     (SawToothGeneratorParams, SawToothGeneratorWrapper, 1.0),
    'sinewave':     (SineWaveGeneratorParams, SineWaveGeneratorWrapper, 1.0),
    'step':         (StepGeneratorParams, StepGeneratorWrapper, 1.0),
    'anomaly':      (AnomalyGeneratorParams, AnomalyGeneratorWrapper, 1.0),
    'spike':        (SpikesGeneratorParams, SpikesGeneratorWrapper, 1.0),
    'ou_process':   (OrnsteinUhlenbeckProcessGeneratorParams, OrnsteinUhlenbeckProcessGeneratorWrapper, 3.0),
}

OUTPUT_BASE = '/workspace/HypOPFN/tempopfn_data'

# Arrow schema (same as TempoPFN)
SCHEMA = pa.schema([
    ("series_id", pa.int64()),
    ("values", pa.list_(pa.list_(pa.float64()))),
    ("length", pa.int32()),
    ("num_channels", pa.int32()),
    ("generator_type", pa.string()),
    ("start", pa.timestamp("ns")),
    ("frequency", pa.string()),
    ("generation_timestamp", pa.timestamp("ns")),
])


def create_wrapper(gen_type, length, seed):
    """Create a TempoPFN generator wrapper."""
    params_cls, wrapper_cls, _ = GENERATOR_CONFIGS[gen_type]

    if gen_type == 'forecast_pfn':
        params = params_cls(
            global_seed=seed, length=length,
            max_absolute_spread=500.0, max_absolute_value=500.0,
        )
    else:
        params = params_cls(global_seed=seed, length=length)

    return wrapper_cls(params)


def generate_batch_data(wrapper, gen_type, batch_size, seed, length):
    """Generate a batch using TempoPFN's wrapper and format as Arrow-compatible dicts."""
    container = wrapper.generate_batch(batch_size=batch_size, seed=seed)
    batch_data = []

    for i in range(container.values.shape[0]):
        values = np.asarray(container.values[i]).flatten().astype(np.float64)
        # Truncate/pad to exact length
        if len(values) > length:
            values = values[:length]
        elif len(values) < length:
            values = np.pad(values, (0, length - len(values)), mode='constant')

        batch_data.append({
            "series_id": seed + i,
            "values": [values.tolist()],  # list of list (univariate = 1 channel)
            "length": length,
            "num_channels": 1,
            "generator_type": gen_type,
            "start": pd.Timestamp(container.start[i]),
            "frequency": container.frequency[i].value,
            "generation_timestamp": pd.Timestamp.now(),
        })

    return batch_data


def write_batch_arrow(batch_data, output_path):
    """Write a batch to Arrow feather file (same format as TempoPFN)."""
    arrays = []
    for field in SCHEMA:
        name = field.name
        if name in ["start", "generation_timestamp"]:
            timestamps = [d[name] for d in batch_data]
            arrays.append(pa.array([t.value for t in timestamps], type=pa.timestamp("ns")))
        else:
            arrays.append(pa.array([d[name] for d in batch_data]))

    table = pa.Table.from_arrays(arrays, schema=SCHEMA)
    feather.write_feather(table, output_path)


def generate_one_type(gen_type, num_batches, length, batch_size, chunk_size=64):
    """Generate all batches for one generator type."""
    output_dir = Path(OUTPUT_BASE) / f'length_{length}' / gen_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing batches
    existing = sorted(output_dir.glob("batch_*.arrow"))
    start_batch = len(existing)
    if start_batch > 0:
        logging.info(f'{gen_type}: Found {start_batch} existing batches, continuing from batch {start_batch}')

    seed_base = hash(gen_type) % (2**31)
    total_series = 0
    start_time = time.time()

    for batch_idx in range(start_batch, start_batch + num_batches):
        batch_seed = (seed_base + batch_idx * batch_size) % (2**32)
        wrapper = create_wrapper(gen_type, length, batch_seed)

        # Generate in chunks to avoid OOM
        all_batch_data = []
        remaining = batch_size
        chunk_seed = batch_seed

        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            try:
                chunk_data = generate_batch_data(wrapper, gen_type, current_chunk, chunk_seed, length)
                all_batch_data.extend(chunk_data)
                remaining -= len(chunk_data)
                chunk_seed += current_chunk
            except Exception as e:
                logging.warning(f'{gen_type} chunk failed: {e}, retrying with new seed...')
                chunk_seed += 1000
                # Recreate wrapper with new seed
                wrapper = create_wrapper(gen_type, length, chunk_seed)
                continue

        # Write batch
        batch_path = output_dir / f'batch_{batch_idx:08d}.arrow'
        write_batch_arrow(all_batch_data, str(batch_path))
        total_series += len(all_batch_data)

        elapsed = time.time() - start_time
        rate = total_series / elapsed if elapsed > 0 else 0
        logging.info(
            f'{gen_type}: batch {batch_idx+1}/{start_batch + num_batches} '
            f'({len(all_batch_data)} series) | '
            f'Total: {total_series:,} | Rate: {rate:.0f}/s | '
            f'File: {batch_path.name} ({os.path.getsize(batch_path)/1e6:.0f}MB)'
        )

        del all_batch_data
        gc.collect()

    return total_series


def main():
    parser = argparse.ArgumentParser(description='Generate TempoPFN native synthetic data')
    parser.add_argument('--types', nargs='+', default=None,
                        choices=list(GENERATOR_CONFIGS.keys()),
                        help='Generator types (default: all)')
    parser.add_argument('--num-batches', type=int, default=3,
                        help='Base number of batches (scaled by weight per generator)')
    parser.add_argument('--batch-size', type=int, default=16384,
                        help='Series per batch file (TempoPFN default: 16384)')
    parser.add_argument('--length', type=int, default=2048,
                        help='Series length (TempoPFN default: 2048)')
    parser.add_argument('--chunk-size', type=int, default=64,
                        help='Generation chunk size (TempoPFN default: 64)')
    args = parser.parse_args()

    gen_types = args.types or list(GENERATOR_CONFIGS.keys())

    print(f'TempoPFN Native Data Generation')
    print(f'================================')
    print(f'Generators: {gen_types}')
    print(f'Base batches: {args.num_batches} (scaled by weight)')
    print(f'Batch size: {args.batch_size:,}')
    print(f'Length: {args.length}')
    print(f'Output: {OUTPUT_BASE}/length_{args.length}/')
    print()

    # Plan
    total_series = 0
    plan = []
    for gen_type in gen_types:
        _, _, weight = GENERATOR_CONFIGS[gen_type]
        n_batches = max(1, int(args.num_batches * weight))
        n_series = n_batches * args.batch_size
        total_series += n_series
        plan.append((gen_type, n_batches, weight))
        print(f'  {gen_type:20s}  weight={weight:.1f}  batches={n_batches:3d}  series={n_series:>8,}')

    print(f'\n  Total planned: {total_series:,} series')
    print()

    # Generate
    overall_start = time.time()
    results = {}
    for gen_type, n_batches, weight in plan:
        try:
            n = generate_one_type(gen_type, n_batches, args.length, args.batch_size, args.chunk_size)
            results[gen_type] = n
        except Exception as e:
            logging.error(f'{gen_type} FAILED: {e}')
            results[gen_type] = 0

    elapsed = time.time() - overall_start

    # Summary
    print(f'\n{"="*60}')
    print(f'Generation Summary ({elapsed:.0f}s total)')
    print(f'{"="*60}')
    grand_total = 0
    for gen_type in gen_types:
        n = results.get(gen_type, 0)
        grand_total += n
        status = '✓' if n > 0 else '✗'
        print(f'  {status} {gen_type:20s}: {n:>8,} series')

    print(f'\n  Grand total: {grand_total:,} series')
    output_dir = Path(OUTPUT_BASE) / f'length_{args.length}'
    print(f'  Output: {output_dir}')

    # Disk usage
    if output_dir.exists():
        total_mb = sum(f.stat().st_size for f in output_dir.rglob('*.arrow')) / 1e6
        print(f'  Disk: {total_mb:.0f}MB')


if __name__ == '__main__':
    main()
