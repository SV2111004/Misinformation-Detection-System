# Member 2 Learning & College Viva Guide: Hybrid RAG & Historical Intelligence

**Student Role:** Member 2 — Hybrid Retrieval-Augmented Generation & Historical Intelligence Lead  
**Domain:** Vector Databases, Information Retrieval, Dense Embeddings, Temporal Propagation Mathematics, Empirical Benchmarking  
**Core Deliverables:** ChromaDB Vector Index, `RAGEngine`, `EvidenceRetriever`, `InfluenceAnalyzer`, Agent B (`EvidenceRAGAgent`), Agent C (`HistoricalInfluenceAgent`), Twitter Snowflake ID Decoder  

---

## 1. Ten-Point Master Learning Syllabus

To master your role and achieve top marks in your college viva examination, thoroughly study these 10 core topics:

1. **Dense Vector Embeddings & `all-MiniLM-L6-v2`:**
   - How semantic text is projected into a 384-dimensional continuous vector space ($\mathbb{R}^{384}$).
   - Sentence-BERT architecture: mean pooling of token representations over BERT/MiniLM layers to create fixed-length sentence embeddings.
   - Why ONNX Runtime (`onnxruntime==1.31.1`) is used locally: achieves rapid vector inference on CPU without external API network latencies or cloud subscription costs.

2. **ChromaDB Persistent Vector Indexing:**
   - Architecture of ChromaDB (`chromadb==1.5.9`) utilizing SQLite for metadata storage and HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor (ANN) search.
   - Vector persistence directory: `dataset/politifact_evidence/chroma_db/`.
   - Collections, document payloads, and metadata schemas (storing `statement_id`, `verdict`, `speaker`, `justification`, `split`).

3. **Cosine Distance vs Cosine Similarity Mathematics:**
   - Dot product of normalized vectors: $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
   - ChromaDB records cosine distance: $d = 1 - \cos(\mathbf{u}, \mathbf{v}) \in [0, 2]$.
   - Converting distance to bounded similarity in $[0.0, 1.0]$:
     $$\text{Score}_{\text{sem}} = \max(0.0, \min(1.0, 1.0 - d))$$

4. **Lexical Sparse Retrieval via TF-IDF:**
   - In-memory `EvidenceRetriever` indexing 10,240 training claims.
   - Sublinear term-frequency scaling ($1 + \log(\text{tf})$) and unigram/bigram n-gram tokenization.
   - Split isolation rules: strictly indexing `split == "train"` to prevent evaluation data leakage from validation or test splits.

5. **Hybrid Score Fusion Algorithm:**
   - Why hybrid search outperforms pure dense or pure sparse retrieval: dense semantic search captures conceptual synonyms (e.g., "job growth" $\leftrightarrow$ "employment gains"), while sparse lexical search captures exact entity names and numeric figures (e.g., "HR 1234", "$500 million").
   - Linear weighted score fusion formula:
     $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{sem}} + (1 - \alpha) \cdot \text{Score}_{\text{lex}}$$
   - Configurable parameter $\alpha \in [0.0, 1.0]$ (default $\alpha = 0.5$).
   - Deterministic tie-breaking on `statement_id`.

6. **LIAR-PLUS Upstream TSV Anomaly Heuristics:**
   - The documented upstream scraping bug in the 2018 LIAR-PLUS dataset where unescaped tabs/newlines shifted columns in $\approx 10-14\%$ of records.
   - Our programmatic alignment verification `check_claim_justification_alignment()`:
     Checks whether speaker surname or informative claim keywords (>4 characters, non-stopwords) appear in the justification text.
   - Setting the boolean flag `has_context_alignment` to alert users and Agent D.

7. **Grounded Summary Synthesis (Deterministic vs LLM):**
   - Analyzing retrieved historical verdicts: counting consensus vs divergence.
   - Deterministic summary template: extracts the top statement ID, speaker, verdict, and hybrid similarity, while explicitly flagging conflicting rulings across citations.
   - Opt-in LLM synthesis: passing citations to Gemini 1.5 Flash or GPT-4o-mini with strict 120-word grounding constraints when API keys are configured.

8. **FakeNewsNet Benchmark Corpus Structure:**
   - 1,056 political news stories from PolitiFact: 432 labeled `fake` and 624 labeled `real`.
   - 583,516 total recorded tweet IDs.
   - Lexical TF-IDF cosine similarity search on headlines with minimum relevance threshold `min_score = 0.08`.

9. **Deterministic 64-bit Twitter Snowflake ID Decoding:**
   - Structure of 64-bit Snowflake IDs introduced by Twitter in November 2010.
   - Bit allocation: 41 bits (timestamp in ms since epoch), 10 bits (machine/datacenter ID), 12 bits (sequence counter).
   - Bit-shift mathematical formula:
     $$\text{Timestamp}_{\text{ms}} = (\text{Tweet\_ID} \gg 22) + 1288834974657$$
   - Epoch constant: November 4, 2010 01:42:54.657 UTC ($1288834974657\text{ ms}$).
   - Three-way ID classification: `valid` ($2010 \le \text{year} \le 2026$), `legacy` ($< 30\text{B}$, pre-Nov 2010), and `malformed`.

10. **Temporal Propagation & Empirical Percentile Mathematics:**
    - First 24-hour volume: counting valid tweets posted within 24 hours of the earliest recorded tweet.
    - Observed propagation span: $(t_{\max} - t_{\min})$ in hours and days.
    - Empirical percentile rank formula:
      $$\text{Percentile}(v \mid C) = \frac{\sum_{i=1}^{|C|} \mathbb{I}(c_i \le v)}{|C|} \times 100\%$$
      calculated independently against fake ($N=432$) and real ($N=624$) benchmark distributions.

---

## 2. Key Code Walkthrough for Member 2

### 2.1 Hybrid Search Fusion (`backend/rag_engine.py:250-392`)
```python
# 1. Candidate pool retrieval: top 20 from Dense and top 20 from Lexical
candidate_k = min(max(top_k * 4, 20), 50)

# Dense search in ChromaDB
chroma_results = self.collection.query(
    query_texts=[text],
    n_results=candidate_k,
    include=["documents", "metadatas", "distances"],
)
# Convert cosine distance to similarity
sem_score = max(0.0, min(1.0, 1.0 - dist))

# Lexical search via TF-IDF
tfidf_results = self.lexical_retriever.search(text, top_k=candidate_k, min_score=0.0)

# Hybrid Fusion
for sid in all_candidate_ids:
    sem_score = semantic_matches.get(sid, {}).get("semantic_score", 0.0)
    lex_score = lexical_matches.get(sid, {}).get("similarity_score", 0.0)
    hybrid_score = round(alpha * sem_score + (1.0 - alpha) * lex_score, 4)
    
    if hybrid_score >= min_score:
        # Check TSV row alignment
        aligned = check_claim_justification_alignment(claim, speaker, justification)
        ranked_results.append({..., "similarity_score": hybrid_score, "has_context_alignment": aligned})

# Deterministic tie-breaking sort
ranked_results.sort(key=lambda item: (-item["similarity_score"], str(item["statement_id"])))
return ranked_results[:top_k]
```

### 2.2 Snowflake Decoding & Temporal Propagation (`backend/influence_analyzer.py:68-173`)
```python
TWITTER_EPOCH_MS = 1288834974657  # Nov 4, 2010 01:42:54.657 UTC

def decode_snowflake_id(tweet_id_str: str) -> Tuple[Optional[datetime], str]:
    if not tweet_id_str.isdigit():
        return None, "malformed"
    num = int(tweet_id_str)
    if num < 30000000000:
        return None, "legacy"
        
    ts_ms = (num >> 22) + TWITTER_EPOCH_MS
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, datetime.timezone.utc)
    if 2010 <= dt.year <= 2026:
        return dt, "valid"
    return None, "legacy"
```

---

## 3. Practical Debugging Recipes for Member 2

1. **How to test ChromaDB retrieval from Python shell:**
   ```powershell
   cd backend
   python -c "from rag_engine import get_rag_engine; e = get_rag_engine(); print('Chroma count:', e.collection.count()); print('Top match:', e.search('unemployment fell', top_k=1))"
   ```
2. **How to test Snowflake ID decoding independently:**
   ```powershell
   python -c "from backend.influence_analyzer import decode_snowflake_id; print(decode_snowflake_id('1178658045934415872'))"
   # Should print: (datetime.datetime(2019, 9, 30, 13, 27, 43, 638000, tzinfo=datetime.timezone.utc), 'valid')
   ```
3. **How to run the 28 tests for RAG and Influence:**
   ```powershell
   pytest tests/test_rag_and_agents.py tests/test_influence_analysis.py -v
   ```

---

## 4. Top 15 College Viva Examination Questions & Answers

### Q1: What is RAG, and why is it necessary for a political misinformation detection system?
**Model Answer:**
"Retrieval-Augmented Generation (RAG) is an architecture that supplements model reasoning with relevant, verifiable facts retrieved dynamically from an authoritative external knowledge base. In political misinformation detection, fine-tuned classifiers are prone to hallucinations, temporal staleness, and superficial linguistic biases. RAG grounds the assessment in published historical PolitiFact fact-checks from the LIAR-PLUS benchmark, providing exact statement IDs, speaker names, and published justifications for every prediction."

### Q2: Why did you combine dense semantic retrieval with sparse lexical retrieval (Hybrid RAG)?
**Model Answer:**
"Dense semantic search (using `all-MiniLM-L6-v2` dense vectors) is exceptional at capturing conceptual similarity, paraphrasing, and semantic intent. However, it can struggle with exact keyword matching, specialized bill acronyms (e.g., 'HR 1234'), and exact numerical figures. Sparse lexical retrieval (using TF-IDF n-grams) is deterministic and excels at exact lexical tokens. By fusing both scores using $\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{sem}} + (1 - \alpha) \cdot \text{Score}_{\text{lex}}$, we obtain the best of both worlds: high semantic recall and high lexical precision."

### Q3: How does ChromaDB store and search vectors?
**Model Answer:**
"ChromaDB uses an underlying SQLite relational database to persist document strings, unique IDs, and metadata dictionaries. For vector indexing, it constructs a Hierarchical Navigable Small World (HNSW) graph. HNSW is a multi-layer graph data structure that performs approximate nearest neighbor (ANN) search with logarithmic time complexity $\mathcal{O}(\log N)$, allowing fast sub-50ms vector queries across thousands of records."

### Q4: Explain the difference between Cosine Distance and Cosine Similarity.
**Model Answer:**
"Cosine similarity measures the cosine of the angle between two vectors:
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
It ranges from $-1$ (opposite) to $+1$ (identical). Cosine distance is defined as $d = 1 - \cos(\theta)$, which ranges from $0$ (identical) to $2$ (opposite). In our `RAGEngine`, we convert ChromaDB's distance metric back into a normalized similarity score using $\max(0.0, \min(1.0, 1.0 - d))$."

### Q5: How do you prevent data leakage during evidence retrieval?
**Model Answer:**
"In our `EvidenceRetriever`, the initialization parameter `include_splits` defaults strictly to `("train",)`. Records from the validation split (`val_evidence.jsonl`) and test split (`test_evidence.jsonl`) are explicitly filtered out during index construction. This guarantees that during offline benchmarking or evaluation on test claims, the retriever cannot access the test split's own ground truth."

### Q6: What is the Twitter Snowflake format, and how does your decoding function work?
**Model Answer:**
"Twitter Snowflake is a 64-bit integer format introduced in November 2010 to generate unique, k-sorted IDs across distributed servers. The most significant 41 bits represent the number of milliseconds elapsed since the custom Twitter epoch (November 4, 2010 01:42:54.657 UTC). Our function `decode_snowflake_id()` performs a bitwise right-shift by 22 bits (`num >> 22`) to extract these 41 bits and adds the epoch offset constant ($1288834974657\text{ ms}$) to reconstruct the exact millisecond UTC creation timestamp."

### Q7: What are 'legacy' IDs in FakeNewsNet, and why can they not be decoded with Snowflake math?
**Model Answer:**
"Before November 2010, Twitter generated sequential integer IDs starting from 1 up to approximately 30 billion. These sequential IDs did not encode a timestamp. Any tweet ID under 30 billion ($< 30,000,000,000$) is categorized as `legacy`. Our decoder safely identifies these legacy IDs, skips the bit-shift calculation, and reports `legacy_ids_count` to prevent integer overflow or bogus dates from corrupting propagation metrics."

### Q8: What propagation metrics do you extract from decoded Snowflake timestamps?
**Model Answer:**
"We extract:
1. **Earliest & Latest Tweet UTC:** The start and end of the observed discussion.
2. **Observed Propagation Span:** Total elapsed duration in hours and days $(t_{\max} - t_{\min})$.
3. **First 24-Hour Velocity:** The count and percentage of valid tweets posted within the first 24 hours of story emergence.
4. **Peak Daily Activity:** The top 5 calendar dates with the highest volume of recorded tweet IDs."

### Q9: What is the mathematical definition of the empirical percentile rank in your analyzer?
**Model Answer:**
"The empirical percentile rank computes the percentage of stories in a reference benchmark category ($C \in \{\text{fake}, \text{real}\}$) that accumulated a tweet volume less than or equal to the observed story's volume $v$:
$$\text{Percentile}(v \mid C) = \frac{\sum_{i=1}^{|C|} \mathbb{I}(c_i \le v)}{|C|} \times 100\%$$
For example, if a matched fake story has 300 tweets and 80% of fake stories in FakeNewsNet have $\le 300$ tweets, it is in the 80th percentile."

### Q10: Does FakeNewsNet record social media reach or follower networks?
**Model Answer:**
"No. The PolitiFact branch of FakeNewsNet contains strictly news headlines, source URLs, PolitiFact labels, and arrays of tweet ID strings. It does NOT contain follower counts, impression numbers, or retweet tree edges. Our system maintains absolute academic honesty by explicitly stating this boundary in the UI and disclaimers, never fabricating unobserved reach."

### Q11: Explain the upstream TSV row-shift issue in LIAR-PLUS and how you handle it.
**Model Answer:**
"When Alhindi et al. published the LIAR-PLUS dataset in 2018, their automated scraping script extracted justifications from PolitiFact HTML. Because some PolitiFact rulings contained unescaped tabs and newlines, the TSV parser experienced row-shifting in $\approx 10-14\%$ of lines, placing a justification into the preceding or succeeding claim's row. Our function `check_claim_justification_alignment()` tests whether the speaker surname or informative keywords from the claim appear in the justification text. Misaligned rows are tagged with `has_context_alignment: false` and surfaced with a warning badge."

### Q12: How does the grounded evidence summary work when no external LLM API key is provided?
**Model Answer:**
"When neither `GEMINI_API_KEY` nor `OPENAI_API_KEY` is provided, `generate_grounded_summary()` executes an objective deterministic fallback. It parses all retrieved records, computes the distribution of historical verdicts (e.g., 3 false, 1 true), flags whether consensus or conflict exists, extracts the top statement ID, speaker, and hybrid similarity score, and formats a factual, neutral summary without hallucination."

### Q13: What is the role of Agent B (`EvidenceRAGAgent`)?
**Model Answer:**
"Agent B encapsulates the hybrid RAG pipeline into a typed agent interface. When invoked with a claim string, it executes `RAGEngine.search()`, generates the grounded summary, checks for consensus or conflict across historical verdicts, formats structured citations, and returns a dictionary with millisecond duration tracking to the orchestrator."

### Q14: What is the role of Agent C (`HistoricalInfluenceAgent`)?
**Model Answer:**
"Agent C encapsulates the FakeNewsNet analysis into a typed agent interface. It searches the 1,056 benchmark stories for headline matches, decodes Twitter Snowflake IDs, calculates temporal propagation metrics, looks up empirical percentiles within both fake and real distributions, and attaches the benchmark limitation disclaimer."

### Q15: How does Member 2's output integrate with Member 3's Verification & Risk Agent (Agent D)?
**Model Answer:**
"Agent D takes the outputs from Agent A (predicted label and confidence), Agent B (retrieved citations and consensus), and Agent C (matched stories and tweet volume). If Agent A predicts *false* and Agent B retrieves citations whose historical PolitiFact verdicts are also *false*, Agent D synthesizes this agreement into `HIGH_MISINFORMATION_RISK`. If they disagree, Agent D flags a discrepancy."

