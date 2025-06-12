# Deep Natural Language Processing  
## Glimpse Framework for Text Summarization  

**Vida Ahmadi**  
**Mozhdeh Hajiani**  

---

This project is a course assignment for Deep NLP, based on the [GLIMPSE](https://github.com/icannos/glimpse-mds) framework:  
**Pragmatically Informative Multi-Document Summarization for Scholarly Reviews**.

We reproduced the original GLIMPSE pipeline on the ICLR 2017 dataset and implemented two key extensions:

### 🔧 Extensions
1. **Domain Adaptation**:  
   Adapted the GLIMPSE framework to biomedical summarization using a PubMed subset to test generalization across domains.

2. **Model Comparison**:  
   Compared performance of different pretrained abstractive summarization models (BART, PEGASUS, and T5) on peer review summarization.

---

For full implementation details, including setup, data processing, summarization generation, and evaluation:  
➡️ **Please see the [`Glimpse_NLP.ipynb`](./Glimpse_NLP.ipynb)** notebook.

## 📁 Data and Models

Due to size constraints, the full PubMed dataset and fine-tuned models are **not included** in this repository.

- ✅ PubMed data subset can be downloaded from [NCBI Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)
- ✅ Trained models and `.pk` candidate files are available upon request


---

> ⚠️ This repository is a student project and is not affiliated with the original authors of GLIMPSE.
