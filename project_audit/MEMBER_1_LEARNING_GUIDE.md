# Member 1 Learning & College Viva Guide: Machine Learning & Claim Intelligence

**Student Role:** Member 1 — Machine Learning Engineer & Claim Intelligence Lead  
**Domain:** Deep Learning Transformers, Natural Language Processing, Statistical Classification, Model Calibration  
**Core Deliverables:** DistilBERT 6-Class Classifier, Baseline Model, Agent A (`ClaimAnalysisAgent`), Confidence Calibration, ML Evaluation Scripts  

---

## 1. Ten-Point Master Learning Syllabus

To master your role and achieve top marks in your college viva examination, thoroughly study these 10 core topics:

1. **Transformer Architecture Fundamentals:**
   - Multi-Head Self-Attention mechanism: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
   - Query ($Q$), Key ($K$), Value ($V$) projections and why dot-product scaling by $\frac{1}{\sqrt{d_k}}$ prevents vanishing gradients.
   - Positional encodings, Layer Normalization, and Residual Connections ($x + \text{Sublayer}(x)$).

2. **DistilBERT Knowledge Distillation:**
   - How DistilBERT achieves a 40% reduction in BERT parameters (66M vs 110M) while retaining 97% of BERT's language comprehension.
   - Triple distillation loss function:
     $$\mathcal{L}_{\text{total}} = \alpha_{\text{ce}} \mathcal{L}_{\text{ce}} + \alpha_{\text{mlm}} \mathcal{L}_{\text{mlm}} + \alpha_{\text{cos}} \mathcal{L}_{\text{cos}}$$
     combining soft cross-entropy with the teacher's logits, masked language modeling loss, and cosine embedding loss.
   - Why token-type embeddings and the pooler layer were removed from DistilBERT for faster CPU inference.

3. **WordPiece Tokenization Mechanics:**
   - Subword tokenization algorithm: handling out-of-vocabulary words using subword prefixes (e.g., `misinformation` $\to$ `mis`, `##information`).
   - Special tokens: `[CLS]` (token classification / sequence representation at position 0) and `[SEP]` (sequence boundary separator).
   - Sequence padding, attention masking (`attention_mask = [1, 1, ..., 0]`), and max length truncation (128 tokens).

4. **The LIAR Benchmark Dataset (Wang, ACL 2017):**
   - 12,836 real-world political statements collected from PolitiFact.
   - Split distribution: Training (10,240), Validation (1,284), Testing (1,267).
   - The 6-class veracity spectrum: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`.
   - The inherent difficulty of 6-class political classification (human PolitiFact fact-checkers debate subtle nuances; state-of-the-art academic benchmarks achieve ~27–30% 6-class accuracy).

5. **Softmax Activation & Confidence Mathematics:**
   - Converting raw model logits $\mathbf{z} = [z_0, z_1, ..., z_5]$ into calibrated probability distribution:
     $$P(Y = i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=0}^{5} e^{z_j}}$$
   - Why $\max_i P(Y=i)$ measures statistical resemblance to LIAR training phrasing, NOT factual reality.
   - The calibration disclaimer: explaining to examiners why high model confidence does not equal verified truth.

6. **Baseline Model Architecture & Purpose:**
   - Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction:
     $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d' \in D : t \in d'\}|}\right) + 1$$
   - Sublinear TF scaling ($1 + \log(\text{tf})$) to dampen the influence of overly frequent political keywords.
   - Multinomial Logistic Regression classifier with L2 regularization.

7. **Agent A (`ClaimAnalysisAgent`) Implementation:**
   - Modular wrapper around DistilBERT in `backend/agents.py`.
   - Integration with PyTorch `eval()` mode and `torch.no_grad()` to eliminate memory graph overhead.
   - Mapping 6 veracity labels to semantic risk categories (`EXTREME`, `HIGH`, `MODERATE`, `LOW`, `MINIMAL`).
   - Millisecond latency tracking and structured output formatting.

8. **Label Mapping & Config Discrepancy Resolution:**
   - Why Hugging Face default training configs export generic labels (`LABEL_0`, `LABEL_1`, ..., `LABEL_5`).
   - The explicit fallback dictionary `ID2LABEL` in `backend/main.py` that maps numeric IDs to canonical LIAR strings:
     `{0: barely-true, 1: false, 2: half-true, 3: mostly-true, 4: pants-fire, 5: true}`.

9. **Evaluation Metrics for Multi-Class Imbalance:**
   - Accuracy vs Macro-Averaged F1 vs Weighted F1.
   - Why Macro F1 ($\frac{1}{6} \sum_{c=1}^{6} F1_c$) is essential when evaluating 6-class political claims: it prevents dominant classes from skewing performance.
   - Precision, Recall, and Confusion Matrix interpretation.

10. **Model Failure Modes & Robustness:**
    - Linguistic negation blindspots (e.g., "The senator did not vote for the bill" vs "The senator voted for the bill").
    - Out-of-vocabulary contemporary entities (e.g., newly elected politicians post-2016).
    - Length truncation: statements exceeding 128 WordPiece tokens lose concluding clauses.

---

## 2. Key Code Walkthrough for Member 1

### 2.1 DistilBERT Inference Engine (`backend/main.py:306-370`)
```python
def predict_distilbert(text: str):
    # 1. Tokenize input with truncation to 128 tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    # 2. Transfer tensors to target device (CPU or CUDA)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 3. Disable gradient calculation to save memory and compute
    with torch.no_grad():
        outputs = model(**inputs)
        # 4. Softmax activation across 6 classes
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()

    # 5. Resolve label mapping fallback
    config_label = model.config.id2label.get(predicted_id)
    if config_label is None or config_label.startswith("LABEL_"):
        label = ID2LABEL.get(predicted_id, str(predicted_id))
    else:
        label = config_label.lower()

    return {
        "model": "DistilBERT",
        "label": label,
        "confidence": float(confidence),
    }
```

### 2.2 Agent A Wrapper (`backend/agents.py:30-100`)
- **Key Method:** `ClaimAnalysisAgent.run(claim: str) -> Dict[str, Any]`
- **Semantic Risk Mapping:**
  ```python
  LIAR_RISK_MAP = {
      "pants-fire": {"risk": "EXTREME", "score": 1.0},
      "false": {"risk": "HIGH", "score": 0.8},
      "barely-true": {"risk": "MODERATE", "score": 0.6},
      "half-true": {"risk": "MODERATE", "score": 0.4},
      "mostly-true": {"risk": "LOW", "score": 0.2},
      "true": {"risk": "MINIMAL", "score": 0.0},
  }
  ```
- **Execution Output:** Returns structured dictionary containing `agent_id`, `status: "completed"`, `duration_ms`, `summary`, `data: {label, confidence, risk_category}`, and `disclaimer`.

---

## 3. Practical Debugging Recipes for Member 1

1. **How to test DistilBERT inference from Python shell:**
   ```powershell
   cd backend
   python -c "from main import predict_distilbert; print(predict_distilbert('Unemployment fell to 4 percent last year.'))"
   ```
2. **How to verify model file existence and weight integrity:**
   ```powershell
   Get-ChildItem -Path models/distilbert_liar_final
   # Should list config.json, model.safetensors (267.8MB), tokenizer.json
   ```
3. **How to run only the ML and Agent A automated tests:**
   ```powershell
   pytest tests/test_rag_and_agents.py -k "agent_a or distilbert" -v
   ```

---

## 4. Top 15 College Viva Examination Questions & Answers

### Q1: Why did you choose DistilBERT instead of standard BERT or RoBERTa?
**Model Answer:**
"DistilBERT reduces the parameter count of BERT by approximately 40% (66 million parameters compared to BERT's 110 million) by stripping the token-type embeddings and the pooler layer while retaining 6 Transformer layers with 12 attention heads. During pre-training, it uses knowledge distillation with a triple loss function to mimic BERT's soft output probabilities, hidden states, and cosine embedding alignment. This delivers 97% of BERT's performance with a 60% faster inference speed on standard CPUs, making it ideal for a real-time web application without requiring expensive GPU infrastructure."

### Q2: What is the significance of the 6-class LIAR dataset taxonomy?
**Model Answer:**
"The LIAR dataset (Wang, ACL 2017) defines a 6-level veracity spectrum: *pants-on-fire*, *false*, *barely-true*, *half-true*, *mostly-true*, and *true*. In political discourse, misinformation is rarely binary black-or-white; political figures frequently speak statements that contain partial truths or cherry-picked statistics with misleading context (*half-true* or *barely-true*). A 6-class formulation captures this nuanced spectrum far better than a crude binary True/False model."

### Q3: Why is 6-class accuracy on the LIAR dataset relatively low (~27-30%) across academic literature?
**Model Answer:**
"A 6-class accuracy of 27–30% is state-of-the-art for short political claims without external evidence. There are two core reasons:
1. Random guessing across 6 balanced classes yields only $\frac{1}{6} \approx 16.7\%$.
2. The claims are evaluated strictly on their linguistic style and phrasing. Distinguishing between *barely-true* and *half-true* or *mostly-true* and *true* requires external historical evidence and context, not just text style. This exact limitation is what motivated us to build the Hybrid RAG engine (Member 2) and Multi-Agent Orchestrator (Member 3) rather than relying on the classifier alone."

### Q4: How does the WordPiece tokenizer handle words it has never seen before?
**Model Answer:**
"WordPiece tokenization decomposes out-of-vocabulary words into smaller known subword units prefixed with `##`. For example, if the word `unconstitutional` is not in the 30,522-token vocabulary, it is tokenized as `['un', '##constitutional']` or `['un', '##constitut', '##ional']`. This eliminates out-of-vocabulary tokens (`<UNK>`) and preserves morphological roots, allowing DistilBERT to represent complex political and legal terminology effectively."

### Q5: What is the mathematical formula for the Softmax function, and why do we use it?
**Model Answer:**
"The softmax function transforms raw unbounded logits $\mathbf{z} \in \mathbb{R}^K$ into a valid categorical probability distribution where each value is in $(0, 1)$ and all values sum to 1:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
We use it because raw logits cannot be directly interpreted as confidences, and the exponentiation naturally highlights the highest-scoring class while preserving relative ranking."

### Q6: Does a 95% model confidence mean the claim is 95% factually true?
**Model Answer:**
"No, absolutely not. Model confidence is strictly a measure of statistical alignment with the linguistic and lexical patterns learned from the training split of the LIAR dataset. A completely fabricated conspiracy theory written in very formal, authoritative language might receive a high model confidence for *mostly-true*. That is why our Agent A outputs an explicit calibration note clarifying that statistical confidence does not equate to verified truth, and why Agent D requires external grounding."

### Q7: Why do you keep the TF-IDF + Logistic Regression baseline in the project?
**Model Answer:**
"In rigorous engineering and academic research, a complex deep learning model must always be benchmarked against an established baseline to justify its computational overhead. Our baseline uses TF-IDF n-grams (1-2 words) with sublinear term-frequency scaling and multinomial Logistic Regression. In our Phase 1 comparison mode, users and examiners can directly see how DistilBERT captures contextual phrasing that bag-of-words representations miss."

### Q8: What does `torch.no_grad()` do during inference?
**Model Answer:**
"`torch.no_grad()` disables PyTorch's dynamic computational autograd engine. Because we are performing inference and not backpropagation, we do not need to track forward activations for gradient computation. This reduces RAM/VRAM consumption by more than 50% and speeds up inference execution."

### Q9: Why is the input truncated to 128 tokens?
**Model Answer:**
"Political claims in the LIAR benchmark are short statements, typically 1 to 3 sentences (average length $\approx 18$ words). Setting `max_length=128` comfortably covers 99.8% of all political claims while capping the quadratic $\mathcal{O}(N^2)$ computational complexity of the Transformer self-attention matrix ($128^2 = 16,384$ operations vs $512^2 = 262,144$ operations)."

### Q10: How does the model handle Hugging Face's `LABEL_0` naming convention?
**Model Answer:**
"When fine-tuning Hugging Face models without explicitly passing an `id2label` dictionary in the training arguments, the classifier head exports default generic names like `LABEL_0`, `LABEL_1`, etc. In `backend/main.py`, our inference function inspects `model.config.id2label`; if it detects `LABEL_`, it safely falls back to our canonical `ID2LABEL` mapping: `0: barely-true, 1: false, 2: half-true, 3: mostly-true, 4: pants-fire, 5: true`."

### Q11: What loss function was used to train the DistilBERT sequence classifier?
**Model Answer:**
"The model was fine-tuned using Categorical Cross-Entropy Loss:
$$\mathcal{L}_{CE} = -\sum_{i=0}^{5} y_i \log(\hat{y}_i)$$
where $\mathbf{y}$ is the one-hot encoded ground truth vector across the 6 LIAR classes and $\mathbf{\hat{y}}$ is the softmax predicted probability vector."

### Q12: Why is Macro F1 preferred over simple Accuracy for model evaluation?
**Model Answer:**
"Macro F1 computes the unweighted arithmetic mean of F1 scores across all 6 classes independently:
$$\text{Macro F1} = \frac{1}{6} \sum_{c=1}^{6} F1_c$$
If a dataset has an uneven distribution of classes (e.g., fewer *pants-fire* instances than *half-true*), a model that performs well on the majority class could achieve decent accuracy while failing on extreme misinformation. Macro F1 treats all veracity classes equally, punishing models that fail on minority classes."

### Q13: What happens if a user submits an empty claim or a 1-character string?
**Model Answer:**
"Both our FastAPI schema validators (`ClaimRequest` with `min_length=1`) and endpoint logic validate input length:
`if not text or len(text) < 3: raise HTTPException(status_code=400)`.
This prevents meaningless tokenization and protects the inference pipeline from processing trivial whitespace."

### Q14: How does DistilBERT represent the entire sentence for classification?
**Model Answer:**
"The input sequence begins with the special token `[CLS]` (Classification). Through the 6 Transformer self-attention layers, the `[CLS]` token's 768-dimensional embedding aggregates contextual representations from all other tokens in the sentence. The final hidden state of the `[CLS]` token is then passed through a linear projection layer and a dropout layer ($p=0.2$) to produce the 6 class logits."

### Q15: How does your Agent A interact with the rest of the multi-agent system?
**Model Answer:**
"Agent A is invoked in Phase 1 concurrently with Agent B (RAG) and Agent C (Influence) by Member 3's `ThreadPoolExecutor`. It executes `predict_distilbert()`, maps the verdict to a benchmark risk category (`EXTREME` down to `MINIMAL`), attaches the confidence calibration disclaimer, and returns a structured trace dictionary to the orchestrator within ~40-60 milliseconds."

