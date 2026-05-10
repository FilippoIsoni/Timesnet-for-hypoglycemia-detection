# Timesnet-for-hypoglycemia-detection
# 🩸 T1D-TimesNet: Advanced 2D-Variation Modeling for Metabolic Safety

[![Dataset: MetaboNet](https://img.shields.io/badge/Dataset-MetaboNet-be123c.svg)](https://arxiv.org/abs/2601.11505)
[![Model: TimesNet](https://img.shields.io/badge/Architecture-TimesNet-0284c7.svg)](https://arxiv.org/abs/2210.02186)
[![Task: Unsupervised Anomaly Detection](https://img.shields.io/badge/Task-Anomaly_Detection-10b981.svg)]()
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-eab308.svg)](https://opensource.org/licenses/MIT)

> **Sistema di Allarme Precoce (Early Warning) puramente non supervisionato per la prevenzione dell'ipoglicemia notturna e il rilevamento di guasti infusionali nel Diabete di Tipo 1, basato sul ripiegamento geometrico fisso delle serie temporali.**

---

## 🎯 1. Executive Summary & Razionale Clinico

La gestione automatizzata del Diabete di Tipo 1 (T1D) si scontra storicamente con la qualità dei dati comportamentali. I task classici di machine learning orientati alla rilevazione dei pasti o dell'attività fisica soffrono di limiti intrinseci legati all'interazione umana: i diari di autocontrollo presentano etichette estremamente sparse, stime dei carboidrati imprecise, omissioni (ambiguità dello zero) e severi ritardi temporali di registrazione.

**T1D-TimesNet** propone un netto cambio di paradigma: ignorare il rumore comportamentale per concentrarsi esclusivamente sulla **fisiologia pura e incontrastata del sonno**. 

Sfruttando la risorsa di popolazione globale **MetaboNet** (campionamento omogeneo a 5 minuti per CGM e record di pompa insulinica), il nostro framework si focalizza su due anomalie critiche oggettive:
1. **Deriva Ipoglicemica Notturna:** Intercettare il calo glicemico indotto da un eccesso di insulina basale con forte anticipo rispetto alle soglie cliniche standard ($< 70\text{ mg/dL}$).
2. **Guasti al Set di Infusione (Pump Failure):** Rilevare l'occlusione silente della cannula identificando la rottura del legame causa-effetto tra l'insulina erogata dal dispositivo e la mancata risposta ipoglicemizzante del corpo.

---

## 🧠 2. Architettura Matematica: La Riformulazione 2D

Le tradizionali reti ricorrenti (RNN) o convoluzionali 1D faticano a mappare dipendenze complesse su sequenze ad altissima dimensionalità. Per superare questo limite, applichiamo i principi di **TimesNet** (ICLR 2023), **rimuovendo la Fast Fourier Transform (FFT)** e imponendo una conoscenza a priori di natura fisiologica: il fortissimo ciclo circadiano umano.

### La Geometria del Tensore (10 Notti $\times$ 96 Step)
Isolando la finestra notturna di 8 ore (es. dalle 23:00 alle 07:00, pari a esattamente $P = 96$ misurazioni a notte), un macro-blocco di osservazione continuo di $N = 10$ notti consecutive ($T = 960$ step totali per variabile) viene mappato in uno spazio bidimensionale tramite un **reshape geometrico fisso**:

$$X_{2D} = \text{Reshape}(X_{1D}) \in \mathbb{R}^{C \times 10 \times 96}$$

Questa topologia dota la matrice di una semantica spaziale rigorosa:
* **Colonne (Variazione Intra-periodo):** I 96 step catturano l'evoluzione metabolica a breve termine interna alla singola notte. L'allineamento è assoluto: la Colonna 0 rappresenta sempre le ore 23:00, la Colonna 48 rappresenta sempre le ore 03:00.
* **Righe (Variazione Inter-periodo):** Le 10 righe allineano verticalmente lo stesso identico istante temporale attraverso i giorni consecutivi, permettendo alla rete di confrontare l'andamento orario sapendo che tra la riga $i$ e la riga $i+1$ intercorrono esattamente 24 ore.
* **Canali ($C$ variabili):** Le feature multi-variate (CGM, Insulina Basale, Insulina Bolo/Totale) vengono impilate nella terza dimensione, venendo elaborate simultaneamente come i canali RGB di un'immagine.

---

## 🛠️ 3. Pipeline Tecnica e Flusso dei Dati

```mermaid
graph TD
    subgraph 1. Preprocessing & Extraction
        A[(MetaboNet Parquet)] -->|Filtro Modalità| B[Solo Soggetti SAP / AID]
        B -->|Estrazione 23:00 - 07:00| C[Imputazione Lineare Gap < 30 min]
    end

    subgraph 2. Augmentation & Reshape
        C -->|Sliding Window: Shift 96 Step| D[Tensori Correlati 1D]
        D -->|Reshape Circadiano Fisso| E[Tensore 3D: C x 10 x 96]
    end

    subgraph 3. Non-Supervised TimesBlock
        E -->|Input| F[Inception Block 2D condiviso]
        F -->|Conv2D Multi-scala 1x1, 3x3| G[Estrazione Pattern Stabilità]
        G -->|Ritorno a 1D + Proiezione| H[Tensore Ricostruito]
    end

    subgraph 4. Early Warning Inference

@article{wolff2026metabonet,
  title={MetaboNet: The Largest Publicly Available Consolidated Dataset for Type 1 Diabetes Management},
  author={Wolff, Miriam K and Aiello, Eleonora Maria and Calhoun, Peter and Qin, Yao and Royston, Sam F},
  journal={Journal of Diabetes Science and Technology},
  year={2026}
}

@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
        H -->|Confronto con Originale| I[Calcolo Loss MSE riga per riga]
        I -->|Se > 95° Percentile Sani| J{🚨 ALLARME PRECOCE}
        J -->|Anticipo Clinico| K[Lead Time: +20/40 min]
    end

    style J fill:#be123c,stroke:#fff,stroke-width:2px,color:#fff
    style E fill:#0284c7,stroke:#fff,stroke-width:2px,color:#fff
   
