# 🧪 Tox21 Toxicity Classifier — ECFP-MLP vs GraphConv

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zqvo04/tox21-toxicity-classifier/blob/main/notebooks/02_ECFP_MLP.ipynb)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![DeepChem](https://img.shields.io/badge/DeepChem-2.7-2ca02c.svg)](https://deepchem.io/)

> **EN** — A multi-label molecular toxicity classifier on the **DeepChem Tox21**
> benchmark, comparing a classical **ECFP fingerprint + MLP** model against a
> **Graph Convolutional Network (GraphConv)**. Built as a cheminformatics +
> deep-learning portfolio project for biotech roles.
>
> **KO** — DeepChem **Tox21** 데이터셋으로 **ECFP 지문 + MLP** 모델과
> **그래프 합성곱 신경망(GraphConv)** 을 비교하는 멀티레이블 독성 분류기.
> 바이오테크 취업용 화학정보학 + 딥러닝 포트폴리오 프로젝트.

---

## 🎯 Overview / 개요

Tox21 은 12개의 독성 엔드포인트(핵수용체 신호 NR-*, 스트레스 반응 SR-*)에 대한
멀티레이블 분류 문제입니다. 레이블에 **결측치(NaN)** 가 많고 클래스 불균형이
심하므로, 본 프로젝트는 **마스킹 손실함수**와 **pos_weight 자동 보정**을 적용합니다.

| 항목 | 내용 |
|------|------|
| Dataset | MoleculeNet **Tox21** CSV (RDKit 로딩, scaffold split) |
| Tasks | 12 multi-label toxicity endpoints |
| Features | **ECFP** (Morgan, radius=2, 2048 bits) / **atom-graph** (RDKit) |
| Models | `ECFPClassifier` (MLP) / `GCNClassifier` (3× GCNConv) |
| Loss | Masked BCE (+pos_weight) / Focal Loss |
| Metrics | per-task ROC-AUC, PR-AUC, F1 + mean ROC-AUC |

---

## 📁 Repository Structure
tox21-toxicity-classifier/
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_ECFP_MLP.ipynb
│ ├── 03_GraphConv.ipynb
│ └── 04_Comparison.ipynb
├── src/
│ ├── dataset.py
│ ├── models.py
│ ├── losses.py
│ ├── train.py
│ └── evaluate.py
├── results/figures/
├── requirements.txt
├── setup_colab.sh
└── README.md

---
## 🚀 Quick Start (Google Colab, T4 GPU)
1. **Open In Colab** 배지를 클릭합니다.
2. 첫 셀에서 환경을 설치합니다:
   ```bash
   !bash setup_colab.sh
순서대로 실행: 01_EDA → 02_ECFP_MLP → 03_GraphConv → 04_Comparison
📊 Performance Comparison
Task	ECFP-MLP ROC-AUC	GraphConv ROC-AUC
NR-AR	0.6946	0.6687
NR-AR-LBD	0.7365	0.7519
NR-AhR	0.8167	0.7919
NR-Aromatase	0.7196	0.7032
NR-ER	0.6142	0.6266
NR-ER-LBD	0.7222	0.7247
NR-PPAR-gamma	0.6754	0.7534
SR-ARE	0.7174	0.7045
SR-ATAD5	0.7145	0.7515
SR-HSE	0.6893	0.7057
SR-MMP	0.7631	0.7662
SR-p53	0.6414	0.7166
Mean	0.7087	0.7221
GraphConv 평균 ROC-AUC 소폭 우세(+0.013). ECFP-MLP 는 NR-AhR 최고 성능(0.817).

🖼️ Results / 결과 이미지
EDA
레이블 통계	물리화학 특성 분포
Label Stats	PhysChem
Label Correlation Heatmap

학습 곡선
ECFP-MLP	GraphConv
ECFP Learning Curve	GCN Learning Curve
ROC Curves
ECFP-MLP	GraphConv
ECFP ROC	GCN ROC
모델 비교
ROC-AUC Bar Chart

Radar Comparison

분자 시각화
독성 분자 예시	SR-MMP False Negatives
Toxic Molecules	False Negatives
🧬 Chemical Interpretation / 화학적 해석 요약
Aromatic nitro / amine — 대사 활성화 → SR-MMP·유전독성 연관
Michael acceptor / epoxide — 단백질·DNA 공유결합 → SR-ARE, SR-p53 활성화
Halogenated aromatic — 친유성 증가 → AhR 수용체 결합
ECFP-MLP 는 부분구조 alert 를 직접 비트 인코딩해 해석성이 높고,
GraphConv 는 메시지 전달로 유연한 표현을 학습하나 해석성은 낮습니다.

🛠️ Tech Stack
RDKit · PyTorch · PyTorch Geometric · scikit-learn · Matplotlib · Seaborn

📄 License
MIT — for educational and portfolio purposes.

---
## 이미지가 표시되는 원리
`![alt text](results/figures/파일명.png)` 문법이 작동하는 이유:
- GitHub는 README를 레포 **루트 기준 상대경로**로 해석합니다
- `results/figures/radar_comparison.png` → 이미 GitHub에 올라간 파일 경로와 정확히 일치
- 별도로 URL을 복사하거나 붙여넣을 필요 없음, 파일이 같은 레포에 있으면 자동으로 표시됨
