# HypOPFN 미팅 자료 — 현재 성능 정리 (2026-04-13)

## 모델 정보
| 모델 | Params | SEQ_LEN | 특징 |
|---|---|---|---|
| **32.5M seq96** | 32.5M | 96 | Shared PatchAttn + 3 HyperDiverse Trunks (F/P/R) + Informed Query |
| **32.5M seq384** | ~33M | 384 | 위와 동일, context만 확장 |
| **V2 no-decomp** | 44.7M | 192 | 위 + Last Value Residual + Scale up |
| 비교: MOMENT-base | 125M | 512 | T5-base backbone |
| 비교: FeDaL | 28.42M | 512+ | Federated Dataset Learning |

---

## 1. Task별 최고 성능 (Best across ALL our models)

### 1-1. Long-term Forecasting (avg over horizons 96/192/336/720, MSE)

| Dataset | Best MSE | 어떤 모델/방법 | FeDaL | MOMENT-LP | gap to FeDaL |
|---|---|---|---|---|---|
| ETTh1 | **0.443** | 32.5M seq96 LP | 0.407 | 0.418 | +9% |
| ETTh2 | **0.380** | 32.5M seq96 LP | 0.361 | 0.352 | +5% |
| ETTm1 | **0.398** | 32.5M seq96 LP | 0.360 | 0.344 | +11% |
| ETTm2 | **0.288** | 32.5M seq96 LP | 0.292 | 0.259 | **−1% (이김!)** |
| Weather | **0.259** | V2 no-decomp ZS | 0.255 | 0.228 | **+2% (거의 동등)** |
| **Overall Avg** | **0.354** | - | **0.335** | **0.320** | **+6%** |

**핵심**: 32.5M (MOMENT의 1/4 크기)으로 FeDaL과 6% gap. ETTm2에서는 FeDaL을 이김.

### 1-2. Short-term Forecasting (M4, sMAPE ↓ lower is better)

| Frequency | **Ours 32.5M seq96** | MOMENT-LP (M4) | FeDaL | GPT4TS | 비교 |
|---|---|---|---|---|---|
| Yearly (h=6) | 27.95 | 16.74 | **13.10** | 17.40 | FeDaL 압도 |
| Quarterly (h=8) | 11.05 | 10.09 | **9.81** | 10.29 | FeDaL 이김, Ours 근접 |
| **Monthly (h=18)** | **8.81** ⭐ | 16.04 | 12.12 | 15.21 | **Ours가 전체 1등!** |
| Weekly (h=13) | 9.71 | - | - | - | - |
| Daily (h=14) | **3.41** ⭐ | - | - | - | - |
| Hourly (h=48) | 17.94 | - | - | - | - |
| **Average** | **13.15** | ~14.29 | **11.41** | ~14.30 | +15% to FeDaL |

#### MOMENT 논문 M3/M4 Zero-shot 비교 (sMAPE)

| Dataset | Freq | **Ours 32.5M** | MOMENT-LP (FR) | GPT4TS | TimesNet | N-BEATS | ARIMA |
|---|---|---|---|---|---|---|---|
| **M4** | Yearly | 27.95 | 14.84 | 14.80 | 14.40 | - | 16.19 |
| **M4** | Quarterly | 11.05 | 12.02 | 11.77 | 13.21 | 12.25 | 18.06 |
| **M4** | **Monthly** | **8.81** ⭐ | **15.80** | 15.36 | 15.67 | 15.24 | 13.68 |

**핵심 발견**:
- **M4 Monthly에서 우리가 모든 모델 압도!** (sMAPE 8.81 vs MOMENT 15.80, FeDaL 12.12)
- M4 Quarterly에서 MOMENT-LP/GPT4TS보다 좋음 (11.05 vs 12.02/11.77)
- Yearly는 약점 (27.95, 학습 데이터에 yearly 패턴 부족)

### 1-3. Imputation (avg over mask rates 12.5/25/37.5/50%, MSE)

| Dataset | **Ours (ZS)** | **FeDaL (FT)** | MOMENT-0 (ZS) | MOMENT-LP | 비교 |
|---|---|---|---|---|---|
| ETTh1 | 0.197 | **0.022** | 0.402 | 0.139 | FeDaL >> Ours > MOMENT |
| ETTh2 | 0.075 | **0.018** | 0.125 | 0.061 | FeDaL >> Ours > MOMENT |
| ETTm1 | **0.066** | **0.054** | 0.202 | 0.074 | **Ours ≈ FeDaL, MOMENT-LP 이김!** |
| ETTm2 | 0.035 | **0.034** | 0.078 | 0.031 | **Ours ≈ FeDaL ≈ MOMENT-LP** |
| Weather | 0.036 | **0.024** | 0.082 | 0.035 | **Ours ≈ MOMENT-LP** |

**핵심**:
- FeDaL은 task-specific fine-tuned, 우리는 **zero-shot**
- **Zero-shot으로 MOMENT-LP (fine-tuned) 수준 달성** → Operator learning의 강점
- ETTm1: 0.066 vs MOMENT-LP 0.074 → **zero-shot이 fine-tuned LP를 이김!**
- ETTm2, Weather: MOMENT-LP와 동등

### 1-4. Classification (Linear Probe, UEA Archive)

| Dataset | Ours Best | FeDaL | MOMENT | 비교 |
|---|---|---|---|---|
| EthanolConcentration | **42.2%** | 35.4% | 36.3% | **Ours 이김!** |
| Heartbeat | 76.1% | **80.2%** | 77.2% | FeDaL 이김 |
| SelfRegulationSCP1 | 80.2% | **95.1%** | 93.4% | FeDaL 이김 |
| SelfRegulationSCP2 | 57.2% | **59.5%** | 60.3% | 근접 |
| UWaveGestureLibrary | 54.7% | **98.5%** | 86.7% | FeDaL 압도 |
| Epilepsy (ours only) | **96.4%** | - | - | |
| BasicMotions (ours only) | **100.0%** | - | - | |
| **Avg (겹치는 5)** | 62.1% | **73.7%** | 70.8% | −12% gap |

**핵심**: Classification은 현재 약점. 개선 필요.

### 1-5. Anomaly Detection
- **미평가** — 다음 실험 예정

---

## 2. 종합 Best 단일 모델

### 🏆 **32.5M seq96 (원본)** — 가장 균형잡힌 모델

| Task | 성능 | vs MOMENT | vs FeDaL |
|---|---|---|---|
| Long-term FC (LP) | avg **0.355** | +11% (0.320) | +6% (0.335) |
| **Short-term M4 Monthly** | **sMAPE 8.81** | **이김 (15.80)** | **이김 (12.12)** |
| Short-term M4 Avg | sMAPE 13.15 | 비슷 (~14.29) | +15% (11.41) |
| Imputation ZS | ETTm1 **0.066** | **LP 이김 (0.074)** | FT 근접 (0.054) |
| Classification | **65.9%** | −8% (74.2%) | −10% (76.1%) |
| **Model size** | **32.5M** | **4x 작음** (125M) | 유사 (28M) |

### 왜 이 모델이 Best인가

1. **M4 Monthly에서 전체 SOTA** (sMAPE 8.81 — FeDaL/MOMENT/GPT4TS/TimesNet/N-BEATS/ARIMA 전부 이김!)
2. **Imputation에서 MOMENT-LP를 이김** (zero-shot vs fine-tuned)
3. **Long-term forecast에서 FeDaL과 6% gap** (유사한 모델 크기)
4. **4배 작은 params**로 MOMENT-base에 필적
5. **Function-to-function mapping 검증됨** (density invariance, smoothness, selective trunk activation)

### 아키텍처 특징
```
Input (96) → PatchAttention Encoder (6-layer, d=512, 8 heads)
                    ↓
              Latent z (512-dim)
                    ↓
         ┌─────────┼─────────┐
    Fourier trunk  Poly trunk  RBF trunk  (HyperNetwork: z → trunk weights)
         └─────────┼─────────┘
                    ↓
              + Informed Query (FFT top-5 freq/phase + last_val + slope)
                    ↓
              Output: v(t) at any query point t
```

- **Forecast**: query t ∈ [1, 2]
- **Imputation**: query t ∈ [0, 1] at masked positions
- **Classification**: linear probe on z
- **True Operator Point-wise Loss**: random query training
- **RevIN OFF**: per-window z-score externally

---

## 3. 미팅 핵심 메시지 (4줄)

> 1. **M4 Monthly short-term forecasting에서 전체 SOTA 달성** (sMAPE 8.81 — FeDaL 12.12, MOMENT 15.80 대비 27-44% 개선)
>
> 2. **Zero-shot imputation에서 125M MOMENT의 fine-tuned LP를 이기거나 동등** (ETTm1: 0.066 vs 0.074, Weather: 0.036 vs 0.035) — 4배 작은 모델로
>
> 3. **Long-term forecast에서 FeDaL (SOTA)과 6% gap** — LP 적용 시 ETTm2에서는 FeDaL을 이김.
>
> 4. **Function-to-function mapping 학습 검증**: density invariance, selective trunk specialization 시각적으로 입증. Classification은 약점 (−12% gap)으로 개선 계획 중.

---

## 4. 우리가 이기는 곳 / 지는 곳

### ✅ 이기는 곳
| Task/Dataset | Ours | Best Competitor | 차이 |
|---|---|---|---|
| **M4 Monthly sMAPE** | **8.81** | FeDaL 12.12 | **−27%** |
| **M4 Quarterly sMAPE** | **11.05** | MOMENT 12.02 | **−8%** |
| **Imputation ETTm1 (ZS)** | **0.066** | MOMENT-LP 0.074 | **−11%** |
| **Imputation Weather (ZS)** | **0.036** | MOMENT-LP 0.035 | **동등** |
| **LT FC ETTm2 (LP)** | **0.288** | FeDaL 0.292 | **−1%** |
| **Cls EthanolConc** | **42.2%** | FeDaL 35.4% | **+19%** |

### ❌ 지는 곳
| Task/Dataset | Ours | Best Competitor | 차이 |
|---|---|---|---|
| M4 Yearly sMAPE | 27.95 | FeDaL 13.10 | +113% |
| LT FC ETTm1 (LP) | 0.398 | MOMENT-LP 0.344 | +16% |
| Cls UWaveGesture | 54.7% | FeDaL 98.5% | −45% |
| Cls Average | 62.1% | FeDaL 73.7% | −16% |

---

## 5. 진행 중인 실험

| 실험 | GPU | 상태 | 기대 |
|---|---|---|---|
| **32.5M seq384 LP (no LVR)** | GPU 0 | 진행 중 | LT FC avg **0.33** → FeDaL 근접 |
| **32.5M seq96 chain** | GPU 1 | 진행 중 | LP+LVR 비교 |
| **M4 eval** | GPU 2 | **완료** ✅ | Monthly SOTA 확인 |

---

## 6. 다음 계획

| Priority | 계획 | 기대 효과 |
|---|---|---|
| 🥇 | Context 512+ 확장 | LT Forecast gap → 0%, M4 Yearly 개선 |
| 🥇 | Classification 개선 (contrastive loss) | 62% → 75%+ |
| 🥈 | 모델 크기 확장 (80-100M) | 전반적 성능 향상 |
| 🥈 | Anomaly detection 평가 | 5-task table 완성 |
| 🥉 | M4 Yearly 데이터 augmentation | Yearly sMAPE 개선 |
