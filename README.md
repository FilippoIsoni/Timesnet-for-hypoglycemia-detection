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
1. **Deriva Ipoglicemica Notturna:** Intercettare il calo glicemico indotto da un eccesso di insulina basale con forte anticipo rispetto alle soglie cliniche standard (< 70 mg/dL).
2. **Guasti al Set di Infusione (Pump Failure):** Rilevare l'occlusione silente della cannula identificando la rottura del legame causa-effetto tra l'insulina erogata dal dispositivo e la mancata risposta ipoglicemizzante del corpo.

---

## 🧠 2. Architettura Matematica: La Riformulazione 2D

Le tradizionali reti ricorrenti (RNN) o convoluzionali 1D faticano a mappare dipendenze complesse su sequenze ad altissima dimensionalità. Per superare questo limite, applichiamo i principi di **TimesNet** (ICLR 2023), **rimuovendo la Fast Fourier Transform (FFT)** e imponendo una conoscenza a priori di natura fisiologica: il fortissimo ciclo circadiano umano.

### La Geometria del Tensore (10 Notti × 96 Step)
Isolando la finestra notturna di 8 ore (es. dalle 23:00 alle 07:00, pari a esattamente 96 misurazioni a notte), un macro-blocco di osservazione continuo di 10 notti consecutive (960 step totali per variabile) viene mappato in uno spazio bidimensionale tramite un **reshape geometrico fisso**:

$$X_{2D} = \text{Reshape}(X_{1D}) \in \mathbb{R}^{C \times 10 \times 96}$$

Questa topologia dota la matrice di una semantica spaziale rigorosa:
* **Colonne (Variazione Intra-periodo):** I 96 step catturano l'evoluzione metabolica a breve termine interna alla sing
