# HypOPFN — 새 서버 인수인계 문서

## 프로젝트 개요

**HypOPFN**: Operator Learning (HyperDeepONet) 기반 Time Series Foundation Model.
- Function-to-function mapping으로 시계열 multi-task (forecast, imputation, classification) 통합
- Query-based output: 임의 시점 t에서 함수값 출력 (하나의 모델로 모든 task)

---

## 현재까지 진행 상황

### 모델 아키텍처 (확정)
```
Input (seq_len) → PatchAttention Encoder (6-layer, d=512, 8 heads)
                          ↓
                    Latent z (512-dim)
                          ↓
              + Informed Query (FFT top-5 freq/phase + last_val + slope)
                          ↓
                ┌─────────┼─────────┐
           Fourier trunk  Poly trunk  RBF trunk  (HyperNetwork: z → trunk weights)
                └─────────┼─────────┘
                          ↓
                   Output v(t) = Σ trunks
```
- **32.5M params** (MOMENT의 1/4 크기)
- **Informed Query**: input의 FFT frequency/phase를 trunk에 주입 → 핵심 component
- **True OP point-wise loss**: random query t에서 MSE
- **RevIN OFF**: per-window z-score externally
- **Multi-task**: forecast (t∈[1,2]) / imputation (t∈[0,1]) / classification (linear probe on z)

### Latent Decomposition (FeDaL-inspired, 실험 예정)
```
z → LatentDecomp → z_trend, z_seasonal
z_trend    → Poly trunk
z_seasonal → Fourier trunk  
z (full)   → RBF trunk (residual)
```
- FeDaL의 Domain Bias Elimination을 operator learning에 적용
- `exp_lotsa_scaling.py`에 `--decomp 1`로 활성화

### 지금까지의 성능 (MEETING_RESULTS.md 참고)

**Best 모델: 32.5M seq96 (checkpoint: full_scale_run.pth)**

| Task | 우리 Best | MOMENT-LP (125M) | FeDaL (28M) |
|---|---|---|---|
| LT Forecast LP avg | **0.355** | 0.320 | 0.335 |
| Imputation ETTm1 (ZS) | **0.066** | 0.074 (LP) | 0.054 (FT) |
| Imputation Weather (ZS) | **0.036** | 0.035 (LP) | 0.024 (FT) |
| M4 Monthly sMAPE | **8.81** | 15.80 | 12.12 |
| Classification avg | 66.4% | 74.2% | 76.1% |

**핵심 발견들:**
1. Imputation에서 MOMENT-LP를 zero-shot으로 이김 (ETTm1, Weather)
2. M4 Monthly에서 모든 모델 압도 (단, 학습 데이터에 M4 포함 — 공정성 이슈)
3. Function-to-function mapping 시각적으로 검증됨 (density invariance, smoothness, trunk specialization)
4. Last Value Residual은 모든 경우에 해로움 → 사용 안 함
5. Input decomposition (L7)은 real data에서 역효과 → latent decomposition으로 전환

### 공정성 이슈 (중요!)
- **현재 학습 데이터 (Pile)에 eval benchmark (ETT, Weather, M4) 포함** → FeDaL 대비 불공정
- **MOMENT도 같은 Pile 사용** → MOMENT 비교는 공정
- **FeDaL은 LOTSA 231B 사용, eval benchmark 명시적 제외** → 가장 공정
- **해결: LOTSA로 학습 전환 (eval 제외)** → 이번 서버에서 할 일!

---

## 이번 서버에서 할 일

### 메인 실험: LOTSA Scaling Study

**목적**: 공정한 비교를 위해 eval benchmark 제외하고 학습 → true zero-shot eval

**실험 설계**:
| Scale | Baseline (no decomp) | + Latent Decomp | 예상 시간 (A100) |
|---|---|---|---|
| 1% | lotsa_s1_base | lotsa_s1_decomp | 각 ~2h |
| 5% | lotsa_s5_base | lotsa_s5_decomp | 각 ~4h |
| 10% | lotsa_s10_base | lotsa_s10_decomp | 각 ~7h |
| 30% | lotsa_s30_base | lotsa_s30_decomp | 각 ~15h |
| 50% | lotsa_s50_base | lotsa_s50_decomp | 각 ~20h |

**제외 데이터 (eval용)**: weather, oikolab_weather, m1_*, m3_*, m4_*
**Synthetic gap filler**: 제외된 패턴 (weather, seasonal, financial, step) 보충

### 실행 순서

```bash
# Step 1: Setup
git clone https://github.com/sheoyon-jhin/HypOPFN.git
cd HypOPFN
bash setup_server.sh   # 패키지 설치 + LOTSA 다운로드 (~20-30분)

# Step 2: 전체 실험 (4 GPU 병렬)
bash run_lotsa_experiments.sh   # 10 experiments, ~40-48h

# 또는 개별 실행
CUDA_VISIBLE_DEVICES=0 python experiments/exp_lotsa_scaling.py --scale 1 --decomp 0 --tag lotsa_s1_base
CUDA_VISIBLE_DEVICES=1 python experiments/exp_lotsa_scaling.py --scale 1 --decomp 1 --tag lotsa_s1_decomp
```

### 필요한 eval 데이터 (별도 다운로드)
LOTSA에서 제외했으므로 eval용으로 별도 필요:
- ETT: `dataset/ETT-small/` (ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv)
- Weather: `dataset/weather/weather.csv`
- Exchange: `dataset/exchange_rate/exchange_rate.csv`
- 다운로드: https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot

### 결과 확인
```bash
# 결과 JSON
cat results/lotsa_s1_base.json
cat results/lotsa_s5_decomp.json
# ...

# 로그
tail -n 20 log/lotsa_s10_base.log
```

---

## 핵심 파일 구조

```
HypOPFN/
├── experiments/
│   ├── exp_lotsa_scaling.py      ★ LOTSA 학습 메인 (이번에 돌릴 것)
│   ├── exp_full_scale_train.py   ★ 32.5M 원본 학습
│   ├── exp_32M_longctx.py          Long context (192/384)
│   ├── exp_full_finetune.py        Full fine-tune
│   ├── eval_full_scale.py          Zero-shot eval
│   ├── eval_full_scale_lp.py       Linear probe eval
│   ├── eval_m4_shortterm.py        M4 short-term eval
│   └── eval_32M_methods.py         All methods eval
├── data/
│   └── download_lotsa.py         ★ LOTSA 다운로드 (eval 제외)
├── data_provider/
│   ├── pile_dataset.py              Pile data loader
│   ├── data_factory.py              Data factory
│   └── data_loader.py               Dataset classes
├── model/
│   ├── DeepONetHyper.py             HyperDeepONet
│   ├── DeepONetHyperMoE.py          MoE variant
│   └── encoders.py                  PatchAttention 등
├── analysis/curriculum/
│   ├── L1~L8 *.py                   Mechanism study
│   └── shared.py                    Shared components
├── for_figure/
│   ├── fig_function_learning.py     Function-to-function 검증
│   └── fig_architecture_compare.py  Architecture 비교
├── run_lotsa_experiments.sh       ★ 4-GPU 병렬 실행
├── setup_server.sh                ★ 서버 초기 세팅
├── MEETING_RESULTS.md               현재 성능 정리
└── HANDOVER_NEW_SERVER.md           이 파일
```

---

## 비교 대상 논문들

| 논문 | Params | Data | 핵심 |
|---|---|---|---|
| **FeDaL** (ICLR 2026) | 28M | LOTSA 231B | FL + DBE/GBE, eval 제외 학습 (가장 공정) |
| **MOMENT** (ICML 2024) | 125M | Pile 1.13B | T5-base, eval 포함 학습 |
| **CauKer** (ICLR 2026) | 1-783M | Synthetic only | GP+SCM 합성, classification 특화 |
| **Chronos** (Amazon) | 8-710M | 84B | In-domain vs zero-shot 구분 |
| **"One-Liners"** (ICLR 2026) | - | - | TSFM anomaly detection은 baseline 못 이김 (분석 논문) |

### FeDaL 상세 (우리의 주요 비교 대상)
- **LOTSA** (Salesforce/lotsa_data on HuggingFace): 231B, 174 datasets
- **DBE**: latent을 trend+seasonal로 분해 → dataset-specific bias 흡수
- **GBE**: gradient correction + core-set tuning
- **ETT, Weather, M4 모두 학습에 미포함** → true zero-shot
- 우리도 같은 LOTSA 사용, 같은 eval benchmark 제외 → 공정 비교

---

## 주의사항

1. **LOTSA 다운로드**: `data/download_lotsa.py`가 eval benchmark 15개를 자동 제외. 수동 변경 불필요.
2. **GPU mapping**: vast.ai에서 GPU 1이 죽는 경우 있었음. `CUDA_VISIBLE_DEVICES` 확인.
3. **Eval 데이터**: LOTSA에 ETT 없음! 별도로 `dataset/ETT-small/`에 넣어야 eval 가능.
4. **Synthetic gap filler**: `exp_lotsa_scaling.py`에 내장. `--synth_ratio 0.3`으로 조절.
5. **Checkpoints**: `checkpoints/` 폴더에 저장. 약 125MB/model.
6. **Results**: `results/` 폴더에 JSON으로 저장. 학습 loss + eval MSE 포함.

---

## 기대 결과

### Data Scaling Curve (Paper Figure 후보)
```
MSE ↓
 |  
 |  ● decomp
 |    ●
 |       ●
 |          ●
 |              ●  
 |  ○ baseline
 |    ○
 |       ○
 |          ○
 |              ○
 └─────────────────── LOTSA Scale (1% → 50%)
```

### 목표 성능 (공정 조건)
| Scale | 기대 avg MSE | vs FeDaL (0.335) |
|---|---|---|
| 1% | 0.45-0.50 | 참고 수준 |
| 5% | 0.40-0.45 | 근접 |
| 10% | 0.38-0.42 | 경쟁력 |
| 30% | 0.35-0.38 | FeDaL 근접 |
| 50% | 0.33-0.36 | **FeDaL 동등/역전** |

### Paper에서 쓸 claim
> "Under fair evaluation (excluding all evaluation benchmarks from pre-training), our 32.5M operator model achieves [X] avg MSE on ETT/Weather with [Y]% of LOTSA, approaching FeDaL (0.335) which uses the full 231B LOTSA dataset."
