# Timesnet-for-hypoglycemia-detection
# 🔬 T1D-TimesNet: Advanced 2D-Variation Modeling for Metabolic Safety

![Header](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## 📋 Executive Summary
**T1D-TimesNet** è un framework di deep learning non supervisionato progettato per la sicurezza dei pazienti con Diabete di Tipo 1. Utilizza l'architettura **TimesNet** (ICLR 2023) per trasformare il monitoraggio continuo del glucosio (CGM) in uno spazio 2D, permettendo il rilevamento precoce di ipoglicemie notturne e guasti tecnici (occlusioni della cannula) con un anticipo clinico significativo.



---

## 🛠 Architettura Matematica: Il Potere del Reshape

Il cuore del progetto risiede nella scomposizione delle variazioni temporali. Per una sequenza di input $X_{1D}$ di lunghezza $T = 960$ (10 notti $\times$ 96 campioni), applichiamo un reshape 2D basato sulla periodicità circadiana $P = 96$:

$$X_{2D} = \text{Reshape}(X_{1D}) \in \mathbb{R}^{C \times 10 \times 96}$$

### Perché questa struttura?
- **Intra-period variations (Colonne):** Catturano la dinamica metabolica rapida (es. la velocità di assorbimento dell'insulina).
- **Inter-period variations (Righe):** Catturano la deriva del fabbisogno basale attraverso i giorni.

```mermaid
graph LR
    subgraph Input_Space
    A[CGM 1D Stream] --> B[Standardization]
    B --> C[Fixed Reshape 10x96]
    end

    subgraph TimesBlock_Core
    C --> D[Inception Block 2D]
    D --> E[Multi-scale Kernels 1x1, 3x3, 5x5]
    E --> F[Parameter Sharing]
    end

    subgraph Output_Space
    F --> G[1D Reconstruction]
    G --> H[MSE Loss Calculation]
    H --> I{Anomaly Score}
    end

    style I fill:#f15,stroke:#333,color:#fff
   
