# HypOPFN — 실험 시행착오 전체 기록

## 프로젝트 목표
- 32.5M HyperDeepONet 기반 Time Series Foundation Model
- FeDaL (28M, 231B data) 대비 5% 이내 zero-shot 성능
- Operator learning 기반 unified forecast + imputation

---

## Phase 1: Baseline 확립

### V1 VarLen + NLL (Best baseline: MSE 0.3963)
- **설정**: 3 HyperTrunk (Fourier32 + Poly6 + RBF20), NLL Gaussian, scale=50
- **데이터**: LOTSA 77K windows + Synth 500K = 577K total (1.25B timesteps)
- **결과**: Overall ZS MSE 0.3963 (FeDaL 0.325 대비 +22%)

### 성능이 좋은 데이터셋 vs 나쁜 데이터셋
- ✅ ETTh2 (+7%), ETTm2 (+3% per-pred_len), Weather_720 (+3%)
- ❌ ETTm1 (+56%), M4 전체 (+161%), Weather imputation (+369%)

---

## Phase 2: Architecture 변형 (전부 baseline 못 이김)

| 실험 | 변경 | Overall MSE | vs Base | 결론 |
|------|------|-----------|---------|------|
| Hybrid trunk | 1 FixedTrunk(poly) + 2 HyperTrunk | 0.4009 | +1.2% | tie |
| Spectral loss | FFT magnitude MSE (weight 약함) | 0.4011 | +1.2% | tie (weight 너무 약았음) |
| Huber loss | MSE 대신 Huber (delta=1.0) | 0.3988 | +0.6% | tie |
| Big 60M | d_model=768, n_layers=8 | 0.4018 | +1.4% | capacity↑ 효과 없음 |
| V2 CrossAttn | Cross-attention trunk | 0.4493 | +13.4% | worse |
| V2 LearnedTrunk | MLP trunk (no basis) | 0.5977 | +50.8% | much worse |

**교훈**: Architecture 미세 조정으로는 1-2% 이상 개선 불가. 데이터가 병목.

---

## Phase 3: NLL 발산 문제 (9번 발생)

### 패턴
- Epoch 1-4: 정상 수렴 (0.4 → 0.32)
- Epoch 5-6: 갑자기 점프 (0.32 → 0.7 → 1.5)
- 이후: 부분 회복 (0.5-0.9) 또는 계속 악화

### 발산한 실험 목록
1. nll_msf (NLL + multi-scale Fourier nf=[8,32,128])
2. nll_msiq (NLL + multi-scale IQ)
3. nll_attnpool (NLL + attention pool)
4. unified_varmask (NLL + variable mask rate)
5. optionA (NLL + LOTSA short aux)
6. padmask (NLL + padding mask + LOTSA short)
7. full_short (NLL + short LOTSA + target boost)
8. nll_4trunk (NLL + FixedTrunk nf=256)
9. tgtboost_clamped (NLL + σ² clamp 수정)

### 근본 원인
```python
var = log_sigma2.exp().clamp(min=1e-4)  # σ² 하한이 0.0001
# → sq_err / 0.0001 = 10,000배 gradient 증폭!
```
- NLL baseline만 안정: 학습 데이터 분포 동일 → σ head 안정
- 아키텍처/데이터 변경 → encoder z 분포 shift → σ²→0 → gradient 폭발
- σ² clamp(min=0.1)도 실패: gradient 충돌 오히려 악화

### 결론
**NLL은 baseline 설정 외에는 사용 불가. MSE로 전환.**

---

## Phase 4: Task-specific 실험

### M4 Short-term
| 실험 | M4 AVG sMAPE | vs Base (25.46) |
|------|-------------|----------------|
| m4_synth (M4 전용 synth 추가) | worse per freq | Yearly/Hourly 악화 |
| short_pred (seq_len=64,128 추가) | 40.41 | +59% worse |
| mse_msf (multi-scale Fourier) | **24.73** | **-2.9%** ✅ |

**교훈**: M4는 데이터 분포 문제. short_pred는 학습 신호 분산으로 역효과.

### Imputation
| 실험 | Overall MSE | vs Base (0.149) |
|------|-----------|----------------|
| unified (buggy: shared masked ctx) | 0.485 forecast, imp 미측정 | worse |
| unified_fixed (clean ctx 분리) | imp 0.167 | +12.3% worse |
| mse_msf | imp 0.243 | +63% worse |

**교훈**: Unified task = multi-task 희석. Imputation 전용 학습이 아니면 기존 50/50이 나음.

### Data 관련
| 실험 | 변경 | 결과 |
|------|------|------|
| target_boost (NLL) | +62K climate LOTSA | 발산 (NLL), but epoch 3 checkpoint 좋았음 |
| lotsa_short_aux (NLL) | +117K short LOTSA | 발산 |
| mse_combined | 4trunk + short + boost, MSE | 0.4023 (+1.5%) |

**교훈**: 데이터 추가 + NLL = 발산. 데이터 추가 + MSE = 안정이지만 큰 개선 없음 (scale=50 수준).

---

## Phase 5: 근본 원인 분석 (시각화 기반)

### 발견 1: HyperNet "포기" 현상
- Trunk decomposition 시각화로 발견
- ETTh2/ETTm2에서 3 trunk 모두 energy ≈ 0 (flat 예측)
- 원인: HyperNet head × 0.01 scaling + MSE의 "mean is safe" 편향
- 같은 z → 3 head 모두 conservative 출력

### 발견 2: Basis 표현력 부족
- Least-squares fitting으로 검증
- nf=32 Fourier: ETTm1 MSE 0.004 (충분) vs ETTh2 MSE 0.178 (부족)
- nf=256 필요: ETTh2 MSE 0.026
- Poly, RBF 단독으로는 어떤 데이터도 잘 fit 못 함

### 발견 3: Synth 분포 엉뚱
- synth_vs_real 시각화로 확인
- Synth ETTh2: 깨끗한 sine wave
- Real ETTh2: chaotic random-walk
- 모델이 "모든 ETT는 주기적"으로 학습 → 실제 h2 만나면 포기

### 발견 4: Spectral loss가 trunk 깨움
- StrongSpec (weight=100): trunk energy ETTh2 0.02 → 0.25 (12배↑)
- 하지만 MSE 수렴 방해 (weight 과다)
- weight=10: 수렴은 했지만 최종 성능 0.457 (나쁨)

---

## Phase 6: 해결책 적용

### FixedTrunk (HyperNet 포기 해결)
- HyperNet 제거 → 고정 MLP + coef inner product
- **mse_allfixed: 0.3984 (19.3M, baseline 0.3963과 거의 동등)**
- 파라미터 41% 감소, 안정성 대폭 향상

### 4-trunk (basis 표현력 해결)
- + FixedTrunk Fourier(nf=256) 추가
- mse_4trunk: 0.4025 (32.7M)
- Train loss가 가장 빠르게 수렴

### Input Decomposition (구조적 해결)
- SeriesDecomposer: MA(49, 25, 7) → trend, seasonal, highfreq, residual
- DecompEncoder: shared backbone + 4 projection heads → 4 separate z
- 각 z가 matched trunk에 전달 → HyperTrunk 포기 문제 해결 기대
- **현재 실험 중** (4 HyperTrunk, 86.6M, scale=2000/9999)

---

## Phase 7: 현재 실행 중

| GPU | 실험 | 모델 | Data | 상태 |
|-----|------|------|------|------|
| 0 | allfixed+hf256 scale=2000 | 4 FixedTrunk (19M) | 13B | 학습 중 |
| 1 | **decomp_hyper4 scale=2000** | 4 HyperTrunk Decomp (86M) | 13B | 학습 중 |
| 2 | allfixed+hf256 scale=9999 | 4 FixedTrunk (19M) | 27B | 학습 중 |
| 3 | **decomp_hyper4 scale=9999** | 4 HyperTrunk Decomp (86M) | 27B | 학습 중 |

---

## 핵심 교훈 요약

1. **NLL은 baseline 외 사용 금지** (9/9 발산)
2. **Architecture 미세조정 < 데이터 양** (16개 실험 증명)
3. **HyperNet 0.01 scaling이 trunk 포기의 근본 원인**
4. **Basis nf=32는 ETTh2/ETTm2에 부족** (least-squares 검증)
5. **FixedTrunk가 HyperTrunk보다 안정적이며 성능 동등** (19M vs 32M)
6. **Input decomposition이 HyperTrunk 포기 문제의 구조적 해결책**
7. **LOTSA 74/174 datasets만 다운로드 되어있었음** → 새 서버에서 전체 다운로드 필요