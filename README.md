# Ticket Classifier: Fine-Tuned LLM on Self-Managed k3s

## Project Summary

Fine-tune a small open-weight LLM to classify customer support tickets and extract structured fields, then serve it on a self-managed k3s cluster with production-grade observability, autoscaling, and load testing.

The goal is to learn the full LLM deployment stack deeply, from training through production serving patterns, and produce a portfolio-worthy repo that demonstrates real infrastructure skills for backend/AI engineering roles.

## Architecture Decisions

### Hybrid training/serving split

Training and serving have different hardware profiles, so we decouple them:

- **Training:** one-time QLoRA fine-tune on a rented NVIDIA GPU (RunPod). This is the industry-standard stack and matches what most AI infrastructure jobs expect.
- **Serving:** local on Apple Silicon via llama.cpp with Metal acceleration, orchestrated by k3s running in Docker/OrbStack.

### Why llama.cpp instead of vLLM

vLLM does not support Apple Silicon (no Metal backend). Rather than pay for continuous cloud GPU time, we use llama.cpp locally, which has:

- Native Metal support with strong throughput on M-series chips
- OpenAI-compatible HTTP server
- Prometheus metrics endpoint
- Same deployment patterns as vLLM (the k8s, observability, and autoscaling work transfers cleanly)

vLLM internals are already well-understood conceptually (PagedAttention, continuous batching, prefix caching), so running it is not required for interview readiness. Demonstrating cross-backend thinking is actually a better fit for hardware-vendor roles like Rebellions, where customers run inference on non-CUDA accelerators.

### Portfolio framing

The README should frame the hybrid setup as deliberate engineering:

> This project demonstrates LLM serving infrastructure across backends. Training uses the standard HuggingFace + trl QLoRA stack on NVIDIA hardware. Serving runs on llama.cpp with Metal acceleration, orchestrated by k3s. The deployment patterns (HPA on queue depth, probe tuning for slow model loads, Prometheus scraping) are backend-agnostic and transfer directly to vLLM or TGI in a CUDA environment.

## Model and Dataset

- **Base model:** Qwen2.5-3B-Instruct. Apache 2.0, strong instruction following at small size, good structured output behavior.
- **Dataset:** [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) on HF Hub. ~27K rows with existing intent/category labels.
- **Augmentation:** use a larger model (GPT-4o-mini or Claude) to generate additional fields per ticket: urgency, sentiment, extracted entities (order_id, product).
- **Target output format (JSON):**

```json
{
  "category": "billing",
  "urgency": "high",
  "entities": {
    "order_id": "12345",
    "product": null
  },
  "sentiment": "frustrated"
}
```

## Budget

Target total cost: under $20.

- RunPod fine-tune session: approximately $5 (one 4090 or A40 pod for 2-4 hours)
- HF Hub: free
- Local serving and k3s: $0
- Buffer for a redo if training fails: ~$10

## Phases

### Phase 1: Data and fine-tuning

1. Fetch Bitext dataset from HF Hub.
2. Augment with urgency, sentiment, and entity extraction using an external LLM API. Save as a versioned JSONL file.
3. Split train/val/test (80/10/10) with stratified sampling on category.
4. Format examples as chat-style conversations targeting the JSON output structure.
5. Rent a RunPod GPU pod (4090 or A40). Install: transformers, peft, trl, bitsandbytes, accelerate.
6. QLoRA fine-tune with trl's SFTTrainer:
   - 4-bit NF4 quantization on base model
   - LoRA rank 16, alpha 32, target all linear layers
   - Cosine LR schedule, warmup, 2-3 epochs
7. Evaluate on held-out test set:
   - Field-level accuracy for category, urgency, sentiment
   - Entity extraction F1
   - Compare against: base Qwen2.5-3B (no fine-tune), GPT-4o-mini baseline
8. Merge adapter into base weights, push merged model to HF Hub (private repo).
9. Tear down RunPod pod.

### Phase 2: Local serving with llama.cpp

1. Convert merged model to GGUF format.
2. Quantize to Q4_K_M for Mac serving (good quality/speed tradeoff).
3. Run llama.cpp server locally, verify OpenAI-compatible endpoint works with `curl`.
4. Enable Prometheus metrics (`--metrics` flag).
5. Containerize: Dockerfile with llama.cpp server, model pulled from HF Hub at startup.
6. Single-container benchmark baseline:
   - Throughput (req/s) at various concurrency levels
   - TTFT p50/p99
   - Inter-token latency
   - Tokens/sec
   - Record these as baseline numbers in the README.

### Phase 3: k3s cluster (local)

1. Install OrbStack (preferred over Docker Desktop on Mac for performance).
2. Install k3d or use k3s directly in a container to create a multi-node local cluster (1 server, 2 agents).
3. Deploy llama.cpp server:
   - `Deployment` with appropriate resource requests/limits
   - `Service` of type ClusterIP
   - `ConfigMap` for server arguments
   - `Secret` for HF Hub token
   - Liveness and readiness probes tuned for slow model load time (startup delay of 30-120s)
4. Install NVIDIA device plugin manifests as a reference (not deployed, since no GPU in cluster). Document how the Deployment would change for a GPU-scheduled pod.
5. Ingress via Traefik (bundled with k3s) so load tests hit a realistic entry point.

### Phase 4: Observability

1. Install kube-prometheus-stack via Helm.
2. ServiceMonitor or PodMonitor for llama.cpp `/metrics`.
3. Grafana dashboard with panels for:
   - Requests per second
   - Queue depth and in-flight requests
   - TTFT and ITL percentiles
   - Tokens generated per second
   - Memory utilization
4. Export dashboard JSON into the repo.

### Phase 5: Autoscaling and production patterns

1. Install prometheus-adapter to expose custom metrics to the k8s API.
2. Configure HorizontalPodAutoscaler on a custom metric (queue depth or requests-in-flight), not CPU. Document why CPU-based HPA is useless for LLM serving.
3. Canary deployment: deploy a second version of the adapter (or a different base model), route a percentage of traffic to it via Service selector weighting or a simple Ingress rule.
4. Graceful shutdown behavior: document how in-flight requests are handled during rolling updates.

### Phase 6: Load testing and writeup

1. k6 or Locust script targeting the Ingress.
2. Ramp test: 1 to 50 concurrent users over 10 minutes.
3. Record where the system saturates, which metric moves first, how autoscaling responds.
4. Capture screenshots of Grafana during load.
5. Final README writeup with:
   - Architecture diagram
   - Fine-tuning results table (base vs fine-tuned vs GPT-4o-mini)
   - Serving benchmark numbers
   - Autoscaling behavior analysis
   - Lessons learned section with concrete operational insights

## Repository Structure

```
ticket-classifier/
├── README.md                      # full writeup with diagrams and metrics
├── data/
│   ├── fetch_bitext.py
│   ├── augment.py                 # uses external LLM to add fields
│   └── splits/                    # train/val/test JSONL (gitignored if large)
├── training/
│   ├── train_qlora.py
│   ├── eval.py                    # compares fine-tuned vs baselines
│   ├── merge_and_push.py
│   └── configs/
├── serving/
│   ├── llamacpp/
│   │   ├── Dockerfile
│   │   └── server_args.md
│   └── vllm-cuda-reference/       # manifests showing GPU-scheduled equivalent
├── k8s/
│   ├── base/                      # Deployment, Service, ConfigMap, Secret
│   ├── ingress/
│   └── kustomization.yaml
├── monitoring/
│   ├── prometheus-values.yaml
│   ├── grafana-dashboard.json
│   └── servicemonitor.yaml
├── autoscaling/
│   ├── prometheus-adapter-values.yaml
│   └── hpa.yaml
├── loadtest/
│   └── k6-ramp.js
└── docs/
    ├── architecture.md
    ├── lessons-learned.md
    └── diagrams/
```

## Key Learning Targets

These are the concrete things to internalize during the build:

- QLoRA mechanics: what the quantized base + LoRA adapter combination actually does at forward-pass time
- GGUF quantization tradeoffs (Q4_K_M vs Q5_K_M vs Q8_0)
- Why LLM server pods need different probe configuration than stateless web services
- How `--ctx-size` (llama.cpp) and `--max-model-len` (vLLM) interact with memory budgets
- GPU scheduling in k8s: device plugins, resource requests, node selection (as reference, not deployed)
- Autoscaling on LLM-specific signals (queue depth, in-flight requests) rather than CPU/memory
- Rolling update behavior when pods take 30-120s to become ready
- Prometheus scraping patterns for long-running inference pods

## Success Criteria

- Fine-tuned model measurably outperforms base Qwen2.5-3B on held-out test set
- Fine-tuned model is competitive with or beats GPT-4o-mini on this narrow task
- Full serving stack runs locally on Mac with one `make up` or equivalent command
- Grafana dashboard shows live metrics during load tests
- HPA visibly scales pod count in response to load
- README tells a clear engineering story that works as a portfolio piece

## Claude Code Handoff Notes

Claude Code should start at Phase 1 unless directed otherwise. Work phase-by-phase rather than trying to scaffold everything at once. Each phase should produce runnable code and a short section of the README before moving on.

Stylistic preferences:
- No em dashes in any documentation or code comments
- Personable, human tone in README and docs
- Do not fabricate results: only record metrics that are actually measured
