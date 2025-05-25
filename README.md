# new_ml_nbs
newest ml prep list 


# study list for ml theory

### List of Recent LLM Topics to Study for ML Interviews

### 1. Transformer Architecture and Variants

- **Key Concepts**:
    - Self-attention and multi-head attention mechanisms.
    - Positional encodings and their evolution (e.g., RoPE, ALiBi).
    - Encoder-decoder vs. decoder-only architectures (e.g., BERT vs. GPT).
    - Sparse Transformers and efficient attention mechanisms (e.g., Performer, Linformer).
- **Recent Advances**:
    - Mixture of Experts (MoE) models for scalability (e.g., Mixtral, DeepSeek).
    - Long-context attention mechanisms (e.g., Transformer-XL, Longformer).
    - Multi-Token Prediction (MTP) for faster training convergence.
- **Why It Matters**: Interviewers expect you to explain Transformer components, compare architectures, and discuss scaling trade-offs. Coding attention from scratch or optimizing it is a common challenge.

### 2. Pretraining and Fine-Tuning Strategies

- **Key Concepts**:
    - Self-supervised learning (e.g., masked language modeling, next-token prediction).
    - Pretraining on massive datasets (e.g., Common Crawl, Wikipedia).
    - Fine-tuning techniques: supervised fine-tuning (SFT), instruction tuning, and parameter-efficient methods (e.g., LoRA, AdapterFusion).
- **Recent Advances**:
    - Direct Preference Optimization (DPO) as an alternative to RLHF for alignment.
    - Curriculum learning for efficient pretraining.
    - Domain-specific fine-tuning (e.g., Med-PaLM for healthcare, ChatLAW for legal).
- **Why It Matters**: Questions often probe how to adapt LLMs for specific tasks, handle data scarcity, or optimize compute. Be ready to code fine-tuning loops or explain overfitting risks.

### 3. Reinforcement Learning from Human Feedback (RLHF) and Alignment

- **Key Concepts**:
    - RLHF pipeline: reward modeling, proximal policy optimization (PPO).
    - Aligning LLMs with human values to reduce harmful outputs.
    - Ethical AI design (e.g., Anthropic’s Constitutional AI).
- **Recent Advances**:
    - Online RLHF for continuous improvement.
    - Alignment without human feedback (e.g., synthetic data-driven alignment).
- **Why It Matters**: Interviewers test your understanding of aligning LLMs safely and efficiently. Expect questions on bias mitigation or coding reward functions.

### 4. Multimodal LLMs

- **Key Concepts**:
    - Integrating text, images, audio, and video (e.g., CLIP, DALL·E, Flamingo).
    - Cross-modal attention and fusion techniques.
    - Multimodal pretraining objectives (e.g., contrastive learning, image-caption alignment).
- **Recent Advances**:
    - GPT-4 and Gemini’s multimodal capabilities for visual question answering.
    - Falcon 2’s text-vision integration.
    - Unified multimodal architectures (e.g., Unified-IO).
- **Why It Matters**: Multimodal LLMs are increasingly relevant for real-world applications. Be prepared to discuss architecture design or code cross-modal embeddings.

### 5. Efficient LLM Training and Inference

- **Key Concepts**:
    - Distributed training (e.g., data parallelism, model parallelism, ZeRO).
    - Quantization and pruning for model compression (e.g., SparseGPT).
    - Mixed-precision training and hardware-aware optimization (e.g., TPUs, GPUs).
- **Recent Advances**:
    - Multi-head Latent Attention (MLA) for reduced compute.
    - FlashAttention for memory-efficient attention.
    - Small Language Models (SLMs) like Mistral-7B for on-device deployment.
- **Why It Matters**: Optimizing LLMs for cost and speed is a hot topic. Expect coding challenges on attention optimization or questions on scaling laws.

### 6. Evaluation Metrics and Benchmarks

- **Key Concepts**:
    - Traditional NLP metrics (BLEU, ROUGE, perplexity).
    - Task-specific metrics (F1 for classification, accuracy for reasoning).
    - Human-centric evaluation (e.g., coherence, factual consistency).
- **Recent Advances**:
    - Benchmarks like MMLU, BIG-Bench, and HELM for reasoning and robustness.
    - Automated evaluation with LLMs as judges.
    - Metrics for hallucination detection and mitigation.
- **Why It Matters**: Interviewers ask how to assess LLM performance or design new metrics. Be ready to code evaluation functions or critique existing benchmarks.

### 7. Hallucination and Bias Mitigation

- **Key Concepts**:
    - Hallucination: LLMs generating factually incorrect outputs.
    - Bias in training data leading to unfair outputs.
    - Techniques like data curation, debiasing, and fact-checking.
- **Recent Advances**:
    - Retrieval-Augmented Generation (RAG) to ground outputs in external data.
    - Self-verification and chain-of-thought reasoning (e.g., DeepSeek-R1).
    - Bias auditing frameworks (e.g., Fairness in LLMs survey).
- **Why It Matters**: Ethical AI is a priority. Expect questions on detecting hallucinations or coding RAG pipelines to improve factual accuracy.

### 8. Agentic LLMs and Tool Use

- **Key Concepts**:
    - LLMs as agents interacting with tools (e.g., code interpreters, web search).
    - Chain-of-thought (CoT) and recursive prompting for reasoning.
    - Task decomposition and planning.
- **Recent Advances**:
    - Agentic Information Retrieval for dynamic data access.
    - OpenAI Agent SDK and DeepSeek’s tool-use frameworks.
    - Symbolic integration for logical reasoning (e.g., Gemini Ultra).
- **Why It Matters**: Agentic LLMs are redefining applications. Be prepared to code a simple agent or explain how LLMs interface with external APIs.

### 9. Prompt Engineering and Decoding Strategies

- **Key Concepts**:
    - Prompt design: zero-shot, few-shot, and chain-of-thought prompting.
    - Decoding methods: greedy, beam search, top-k, top-p sampling.
    - Temperature and stopping criteria.
- **Recent Advances**:
    - Automated prompt optimization (e.g., PromptBreeder).
    - Structured prompting for complex tasks (e.g., ReAct).
    - Adaptive decoding for context-aware generation.
- **Why It Matters**: Prompt engineering is a practical skill tested in interviews. Expect to write prompts or code decoding algorithms.

### 10. Emerging Architectures Beyond Transformers

- **Key Concepts**:
    - Limitations of Transformers (e.g., quadratic complexity, context length).
    - Recurrent architectures like SSM (State Space Models) and Mamba.
    - Hybrid models combining attention and convolution.
- **Recent Advances**:
    - Mamba’s linear scaling for long sequences.
    - RetNet for efficient sequential processing.
    - Test-time training layers for dynamic adaptation.
- **Why It Matters**: Interviewers may ask about Transformer limitations or emerging alternatives. Be ready to compare architectures or discuss scalability.

### 11. Security and Privacy in LLMs

- **Key Concepts**:
    - Data leakage risks in training datasets.
    - Adversarial attacks (e.g., prompt injection).
    - Differential privacy and federated learning.
- **Recent Advances**:
    - Privacy-preserving fine-tuning (e.g., DP-SGD).
    - SafeguardGPT’s alignment pipeline.
    - Secure multi-party computation for LLM deployment.
- **Why It Matters**: Security is critical for enterprise LLMs. Expect questions on mitigating risks or coding privacy-preserving algorithms.

### 12. Grokking and Generalization

- **Key Concepts**:
    - Grokking: sudden performance jumps after extended training.
    - Double descent phenomenon in overparameterized models.
    - Generalization to out-of-distribution tasks.
- **Recent Advances**:
    - Studies linking grokking to double descent.
    - Cross-modal generalization (e.g., English math to French math).
    - Theoretical models for emergent capabilities.
- **Why It Matters**: Understanding why LLMs generalize is a deep theoretical topic. Be prepared to explain grokking or design experiments to study it.
