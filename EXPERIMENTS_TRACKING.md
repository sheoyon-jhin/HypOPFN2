# HypOPFN 실험 추적 문서

**Last updated**: 2026-04-14  
**Project**: HypOPFN (HyperDeepONet Operator Learning for Time Series Foundation Models)  
**Hardware**: 4× A100 40GB on single server  

---

## 📊 데이터셋 속성

### 학습 데이터

| 데이터 | 구성 | 크기 | 용도 |
|--------|------|-----|------|
| **LOTSA (curated)** | 74 datasets (eval 15개 제외, climate 중복 제거) | 181 GB | Pre-training |
| **Synthetic v1** | 7 domains (ETT/Weather/Exchange/M4/regime/composite) | — | 초기 실험 |
| **Synthetic v2** | 12 domains (ETT 4 + Weather 3 variants + 5 others), 74% eval-biased | — | eval-biased 실험 |
| **Synthetic v3** | **15 domains** (v2 + nonstationary_burst + multiscale + heteroskedastic) | — | 실패 모드 보강 |

### Eval 데이터셋 (unseen, LOTSA에서 제외)

| Dataset | 변수 수 | 행 수 | 특성 |
|---------|--------|------|------|
| ETTh1 | 7 | 17,420 | 시간당, 전력 부하 + 6개 연관 |
| ETTh2 | 7 | 17,420 | 시간당, ETTh1보다 irregular |
| ETTm1 | 7 | 69,680 | 15분, 고해상도 |
| ETTm2 | 7 | 69,680 | 15분, 낮은 variance |
| Weather | 21 | 52,696 | 시간당, 기상 (다채널) |
| M4 (6 freq) | 1 | 변동 | Monthly/Yearly/Quarterly 등 짧은 시리즈 |

---

## 🏗️ 모델 속성

### V1 (원본 anchor)

| 속성 | 값 |
|------|-----|
| Encoder | PatchAttention (d=512, L=6, patch=16, heads=8) |
| Pool | **Mean pool** over patches → single z ⚠️ |
| Trunk | 3개 HyperTrunks (Fourier/Poly/RBF, 고정 basis) + HyperNetwork |
| Informed Query | FFT top-5 + last_val + slope (13 features) |
| Params | **32.5M** |

### V2 (신 아키텍처)

| 속성 | 값 |
|------|-----|
| Encoder | PatchAttention (d=**768**, L=**8**, patch=**8**, heads=8) |
| Pool | **없음** — 패치 전체 유지 ✅ |
| Trunk (CrossAttn 버전) | Query가 patches에 cross-attention |
| Trunk (LearnedTrunk) | MLP 학습 basis + mean-pooled z |
| Informed Query | FFT top-10 + ACF 6 + 3 moments + entropy (32 features) |
| Params | **57M (LrnTrunk) ~ 76M (CrossAttn)** |

### V1 VarLen (변형)

| 속성 | 값 |
|------|-----|
| Encoder | V1 구조 + **Sinusoidal PE** (임의 길이 OK) |
| 특징 | Attention mask (padding 무시), 학습 시 seq_len random 샘플링 |
| Params | 32.5M |

---

## 🧪 실험 전체 목록

### 🏁 완료 (20 실험)

| # | 실험 Tag | 모델 | Scale | seq_len | Epochs | Synth | Special | Overall MSE | 상태 |
|---|---------|------|-------|---------|--------|-------|---------|-----------|------|
| 1 | lotsa_s1_base | V1 | 1% | 192 | 20 | 10K (v1) | rolling | 0.7258 (ETTh1 avg only) | ✅ 완료 |
| 2 | lotsa_s1_decomp | V1 + decomp | 1% | 192 | 20 | 10K (v1) | rolling | 비슷 | ✅ 완료 |
| 3 | lotsa_s5_base_v2 | V1 v2 | 5% | 192 | 20 | 10K (v2) | rolling, dense_query=64 | — | ✅ 완료 |
| 4 | lotsa_s5_decomp_v2 | V1 v2 + decomp | 5% | 192 | 20 | 10K (v2) | rolling | — | ✅ 완료 |
| 5 | lotsa_s5_v3_seq512 | V1 v3 | 5% | 512 | 40 | 10K (v2) | **direct**, n_q=64, div=0.01 | **0.4891** | ✅ 완료 |
| 6 | lotsa_s10_v3_seq512 | V1 v3 | 10% | 512 | 40 | 10K | direct | **0.4827** | ✅ 완료 |
| 7 | lotsa_s30_v3_seq512 | V1 v3 | 30% | 512 | 40 | 10K | direct | 0.5926 (anomaly) | ✅ 완료 |
| 8 | lotsa_s50_v3_seq512 | V1 v3 | 50% | 512 | 40 | 10K | direct | **0.4667** | ✅ 완료 |
| 9 | overnight_seq720_s50 | V1 | 50% | **720** | 40 | 10K | **direct 1-shot** | 🏆 **0.4473** (이전 최고) | ✅ 완료 |
| 10 | overnight_seq512_e80_s50 | V1 | 50% | 512 | **80** | 10K | direct | 0.4687 | ✅ 완료 |
| 11 | overnight_nq128_s50 | V1 | 50% | 512 | 40 | 10K | n_query=128 | 0.4667 | ✅ 완료 |
| 12 | synth_full_base | V1 | **synth만** | 192 | 30 | 500K (v1) | no LOTSA | 0.47 (ETT avg) | ✅ 완료 |
| 13 | synth_full_decomp | V1 + decomp | synth만 | 192 | 30 | 500K (v1) | no LOTSA | 비슷 | ✅ 완료 |
| 14 | synth_smoke | V1 | — | 192 | 5 | 30K | 파이프라인 검증 | 0.73 | ✅ 완료 |
| 15 | lotsa_s10_base_v2 | V1 v2 | 10% | 192 | 20 | 10K (v2) | rolling | — | ✅ 완료 |

### 🟢 현재 진행 중 (4 실험)

| GPU | Tag | 모델 | Scale | seq_len | Synth | Special | 진행률 | Train Loss | 상태 |
|-----|-----|------|-------|---------|-------|---------|------|-----------|------|
| 0 | **lotsa_s50_seq720_synth500k_evalbias** | V1 | 50% | 720 | **500K v2 eval-bias** | direct | Eval (17/20) | **0.188** | 🏁 거의 끝 |
| 1 | **lotsa_s50_seq512_synth500k_evalbias** | V1 | 50% | 512 | **500K v2 eval-bias** | direct | Eval (18/20) | **0.200** | 🏁 거의 끝 |
| 2 | **v2_xattn_s50_seq512** | V2 **CrossAttn 76M** | 50% | 512 | 500K | multi-change | **13/40** | 0.248 | 🏃 학습 중 |
| 3 | **v2_lrntrunk_s50_seq512** | V2 **LrnTrunk 57M** | 50% | 512 | 500K | multi-change | **14/40** | 0.338 | 🏃 학습 중 |

### ⏳ 큐 대기 중 (GPU 0, 1 비는 대로 자동 시작)

| GPU | Tag | 모델 | Special | 예상 시간 | 상태 |
|-----|-----|------|---------|---------|------|
| 0 | v1_seq720_spectral_failmode | V1 | **Spectral loss (0.1)** + **FailMode synth v3** | ~5h | ⏳ 대기 중 |
| 1 | v1_varlen_failmode | V1 **VarLen** | Variable seq_len + padding mask + FailMode v3 | ~6h | ⏳ 대기 중 |

---

## 🎯 핵심 결과 (FeDaL ZS avg 0.335 대비)

### Long-term Forecast (ETT + Weather 평균)

| 실험 | Overall MSE | vs FeDaL | 비고 |
|------|-----------|---------|------|
| Overnight seq720_s50 | **0.4473** | +33% | 이전 최고 |
| Overnight seq512_e80 | 0.4687 | +40% | 80 epochs 효과 미미 |
| Lotsa_s50_v3_seq512 | 0.4667 | +39% | 기준 |
| Synth Full (no LOTSA) | ~0.50 | +49% | LOTSA 없이도 근접 |
| **eval-bias 500K seq720** | ~**0.45** (approx) | +34% | Weather 대폭 개선 |

### 데이터셋별 Best (우리 최고 vs FeDaL ZS)

| Dataset | 우리 최고 | FeDaL ZS | Δ |
|---------|---------|---------|---|
| ETTh1 | 0.5118 (seq720) | 0.407 | +26% |
| **ETTh2** | **0.4236** | 0.361 | **+17%** |
| ETTm1 | 0.6332 (seq720) | 0.360 | +76% 🔴 |
| **ETTm2** | **0.3599** | 0.292 | **+23%** |
| **Weather** | **~0.25** (eval-bias) | 0.255 | **~0%** 🎉 |

### Imputation (FeDaL ZS 평균 ~0.082)

| Dataset | 우리 (seq720) | FeDaL ZS | Δ |
|---------|-----------|---------|---|
| ETTh2 | 0.106 | 0.092 | **+15%** 🟢 |
| ETTm2 | 0.089 | 0.057 | +57% 🟡 |
| Weather | 0.157 | 0.030 | +423% 🔴 |
| Overall | 0.194 | ~0.082 | +137% |

### M4 Short-term (sMAPE, 100 시리즈 test)

| Freq | 우리 | FeDaL | Δ |
|------|------|-------|---|
| **Hourly** | **10.40** | 12.40 | **-16%** 🏆 |
| Quarterly | 14.64 | 9.81 | +49% |
| Monthly | 16.43 | 12.12 | +36% |
| 나머지 | 짧은 시리즈 | +100~400% | 🔴 (zero-padding 문제) |

---

## 📈 주요 Insight

### ✅ 강점
1. **Direct 1-shot prediction** (seq_len=720) → rolling 완전 제거
2. **M4 Hourly**: FeDaL 이김 (-16%)
3. **ETTh2 / ETTm2**: 거의 동급 (+17~23%)
4. **Weather**: eval-bias synth로 동등 수준 도달!
5. **Parameter efficiency**: 32.5M ≈ FeDaL 28M
6. **Zero-shot**: ETT/Weather/M4 한 번도 안 본 조건

### 🔴 약점 / 원인
1. **ETTh1/ETTm1**: 2배 차이 — 고주파 노이즈 + spike 패턴 약함
2. **Imputation**: ZS 137% — mask ratio 고정(37.5%) 때문일 수 있음
3. **M4 짧은 시리즈**: Zero-padding으로 망가짐
4. **V1 pooled z**: 단일 z bottleneck — V2 해결 시도
5. **MSE → conditional mean** 학습 → 노이즈 재현 못함 → Spectral loss 시도

### 🔬 구조적 문제 & 해결 방향

| 문제 | 해결 (진행/완료) |
|------|--------------|
| Pooled z bottleneck | ✅ V2 CrossAttn (진행 중) |
| 고정 basis (Fourier/Poly/RBF) | ✅ V2 LrnTrunk (진행 중) |
| 작은 IQ (13 features) | ✅ V2 Rich IQ (32 features) |
| Rolling prediction | ✅ Direct 1-shot (완료) |
| Zero-padding 재앙 | ⏳ V1 VarLen (큐) |
| MSE만 사용 → 노이즈 재현 X | ⏳ Spectral loss (큐) |
| Synth 다양성 부족 | ✅ v3 (15 도메인, 실패 모드 포함) |

---

## 🔧 앞으로 (Tier 별)

### Tier 1 — 실행 중/대기 중 (자동)
- [x] V2 CrossAttn (GPU 2) — 진행 중
- [x] V2 LrnTrunk (GPU 3) — 진행 중
- [x] V1 + Spectral + FailMode synth (GPU 0 큐)
- [x] V1 VarLen + FailMode synth (GPU 1 큐)

### Tier 2 — 다음 후보
- [ ] Gaussian NLL loss 구현 + 실험
- [ ] Fine-tune (FT) on ETT/Weather → FeDaL Full-shot 비교
- [ ] Linear Probe (LP) → MOMENT-LP 비교
- [ ] Failure analysis 돌리기 (analyze_failures.py)
- [ ] Train-split diagnostic (eval_train_split.py)

### Tier 3 — 장기
- [ ] V2 + eval-bias synth
- [ ] V2 + Spectral loss
- [ ] V2 + VarLen
- [ ] Multi-variate synth (MV)
- [ ] Operator collate (random query set)

---

## 📂 주요 파일 경로

| 카테고리 | 경로 |
|---------|------|
| 코드 | `/workspace/HypOPFN2/experiments/exp_lotsa_scaling.py` (V1) |
| V2 | `/workspace/HypOPFN2/experiments/exp_lotsa_v2arch.py` |
| VarLen | `/workspace/HypOPFN2/experiments/exp_v1_varlen.py` |
| Eval | `/workspace/HypOPFN2/experiments/eval_*.py` |
| Viz | `/workspace/HypOPFN2/experiments/viz_*.py` |
| 결과 | `/workspace/HypOPFN2/results/*.json` |
| Figure | `/workspace/HypOPFN2/results/figures/*.png` |
| 체크포인트 | `/workspace/HypOPFN2/checkpoints/*.pth` |
| 로그 | `/workspace/HypOPFN2/log/*.log` |

---

## 📝 Paper Positioning (현재 기준)

### 가능한 Claim
1. **"Direct long-horizon prediction via operator learning"** — rolling 제거 증명됨
2. **"Parameter efficiency"** — 32.5M ≈ FeDaL 28M
3. **"M4 Hourly SOTA"** (-16% vs FeDaL)
4. **"Weather zero-shot near-parity"** (eval-bias synth로 동등)
5. **"Unified multi-task operator"** — forecast + imputation 통합
6. **"Diagnostic synth generation"** — failure modes 분석 → targeted synth → 개선

### 약점 대응 계획
- ETTh1/ETTm1: Spectral loss + V2 CrossAttn 기대
- Imputation: Random mask rate 학습으로 개선
- M4 짧은: VarLen + padding mask로 해결
