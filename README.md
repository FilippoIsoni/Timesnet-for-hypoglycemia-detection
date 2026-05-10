# Timesnet-for-hypoglycemia-detection
# 🩸 T1D-TimesNet: Nocturnal Hypoglycemia Early Warning

[![MetaboNet Dataset](https://img.shields.io/badge/Dataset-MetaboNet-red.svg)](https://arxiv.org/abs/2601.11505)
[![Architecture-TimesNet](https://img.shields.io/badge/Architecture-TimesNet-blue.svg)](https://arxiv.org/abs/2210.02186)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)

> **Rilevamento intelligente e non supervisionato delle anomalie glicemiche notturne tramite modellazione 2D delle variazioni temporali.**

---

## 🎯 Il Problema: Rumore vs Fisiologia

[span_0](start_span)Il monitoraggio del Diabete di Tipo 1 (T1D) è spesso ostacolato da dati "sporchi": diari dei pasti incompleti, stime errate dei carboidrati e registrazioni tardive dell'attività fisica[span_0](end_span). 

**T1D-TimesNet** aggira questo rumore focalizzandosi sulla **fisiologia pura del sonno**. Durante la notte, l'assenza di interferenze esterne rende ogni deviazione un segnale critico. Il nostro obiettivo è prevedere l'ipoglicemia prima che avvenga, analizzando la rottura della stabilità metabolica.

---

## 🧠 La Soluzione: Temporal 2D-Variation

Invece di analizzare i dati come una sequenza 1D piatta, utilizziamo un'architettura **TimesNet adattata**. Trasformiamo il segnale in un **Tensore 2D** dove:
- **Colonne (Phase):** Rappresentano i 96 step (5 min ciascuno) delle 8 ore notturne.
- **Righe (History):** Rappresentano una sequenza di 10 notti consecutive.

[span_1](start_span)In questo modo, i kernel convoluzionali dell'Inception Block possono confrontare simultaneamente l'evoluzione oraria (intra-notte) e il trend storico (inter-notte)[span_1](end_span).

---

## 🛠 Pipeline di Elaborazione

```mermaid
graph TD
    A[MetaboNet Raw Data] -->|Filtro SAP/AID| B(Preprocessing)
    B -->|Sliding Window 1 Notte| C{Dataset Augmentation}
    C -->|Reshape 10x96| D[Tensore 2D Multi-variato]
    D --> E[TimesBlock Inception]
    E -->|Ricostruzione| F[Errore di Ricostruzione MSE]
    F -->|Superamento Soglia| G{🚨 ALLARME PRECOCE}
    
    style G fill:#f96,stroke:#333,stroke-width:2px
    style A fill:#fff,stroke:#333
