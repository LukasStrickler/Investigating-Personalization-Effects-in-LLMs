# Free AI API Providers for LiteLLM

> **Living Document** | Last Updated: March 2026
> 
> This document catalogs **perpetually free / rate-limited AI API tiers** compatible with LiteLLM. We focus exclusively on providers offering ongoing free access—not one-time credits or time-limited trials.

---

## What Qualifies as "Free Tier"

### Mandatory Requirements

All providers MUST meet these criteria:

| Requirement | Description |
|-------------|-------------|
| **API Access** | Actual REST API endpoint (not just web playground) |
| **LiteLLM Compatible** | Native LiteLLM provider OR OpenAI-compatible endpoint |
| **Perpetual Free** | Ongoing free access—not trials, not one-time credits |
| **Text In / Text Out** | Must support normal chat-style text input and text output |

### Inclusion Criteria

| ✅ Included | ❌ Excluded |
|------------|-------------|
| Unlimited rate-limited access | One-time credits ($5, $25, etc.) |
| Daily/monthly recurring quotas | Time-limited trials (30-90 days) |
| Token quotas that reset | Pay-per-use with no free allowance |
| Completely free models | Freemium with steep paywalls |
| REST API with programmatic access | Web-only interfaces |
| General-purpose multimodal chat models with text input/output | Specialized OCR / speech / image-gen / video-gen / embedding / rerank / computer-use-only endpoints |

**Key metric**: Can you run experiments indefinitely via API without paying? If yes → included.
---

## Provider Rankings

> **Note**: This overview is the authoritative section. Providers marked "dashboard-only" do not publish static rate limits; live values are visible in their developer consoles. See individual provider sections for scope details (per-model vs per-provider vs account-level).

### By Overall Value (Quality × Quota)

| Rank | Provider | Quota | Best Model | Quality | Requests/Day | LiteLLM |
|------|----------|-------|------------|---------|--------------|---------|
| 1 | **Groq** | Unlimited | Llama 3.3 70B | ★★★★★ | ~1,000 | `groq/` |
| 2 | **Google Gemini** | Unlimited | Gemini 2.5 Pro | ★★★★★ | Dashboard | `gemini/` |
| 3 | **NVIDIA NIM** | Dev member access | Llama 3.1 405B | ★★★★★ | Dashboard | `nvidia_nim/` |
| 4 | **Cerebras** | 1M tok/day | Llama 3.1 8B | ★★★★ | ~500 | `cerebras/` |
| 5 | **OpenRouter** | 50/day free, 1K/day paid | Qwen3 Coder / Hermes 405B | ★★★★ | Shared quota | `openrouter/` |
| 6 | **Mistral** | Dashboard | Mistral Small | ★★★★★ | Dashboard | `mistral/` |
| 7 | **Silicon Flow** | 12 free chat models | Qwen/GLM/DeepSeek | ★★★★ | Dashboard | `openai/` |
| 8 | **Zhipu AI** | 3 free chat models | GLM-4.x Flash | ★★★★ | Dashboard | `zai/` |
| 9 | **Cohere** | 1K calls/month | Command A | ★★★★ | ~33/day | `cohere_chat/` |
| 10 | **GitHub Models** | Preview | DeepSeek-R1 / GPT-5 | ★★★★ | 8-450/day by tier | `github/` |
| 11 | **Cloudflare** | 10K neurons/day | Llama 3.2 | ★★★ | ~190-1.3K | `openai/` |

---

## Universal Model-Provider Matrix

> Rows = selected validated chat models (alphabetical) | Columns = Providers | Cells = Availability + Rate Limits
>
> This matrix is a curated comparison layer, not a full vendor catalog. It includes general-purpose chat / reasoning models that support normal text input and text output, including eligible multimodal models. Specialized OCR, speech, image-generation, video-generation, embedding, rerank, and computer-use-only endpoints are still excluded.

### Legend

**Cell format**: `VALUE (SCOPE)`
- `(M)` = model-specific quota
- `(S)` = one shared pool across models / account / workspace / task
- `(D)` = dashboard-only / plan-dependent / live console value

**Value labels**:
- `Unlimited` = no published hard daily cap; still rate-limited
- `FREE` = model is listed as free; exact live quota is not cleanly published
- `Dashboard` = check provider console for the live limit
- `Preview` = available, but GitHub/plan tier decides the actual quota
- `—` = not available

### Model Availability Matrix

| Model | Groq | Gemini | NVIDIA NIM | Cerebras | OpenRouter | Mistral | Silicon Flow | Zhipu | Cohere | GitHub | Cloudflare |
|-------|------|--------|-------------|----------|------------|---------|--------------|-------|--------|--------|------------|
| **Command A** | — | — | — | — | — | — | — | — | 20 RPM (M) | — | — |
| **DeepSeek-R1** | — | — | — | — | — | — | — | — | — | 8-12/day (D) | — |
| **DeepSeek-R1-0528** | — | — | — | — | — | — | — | — | — | 8-12/day (D) | — |
| **DeepSeek-R1-Distill-Qwen-7B** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **Dolphin-Mistral-24B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Gemma-3-4B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Gemma-3-12B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Gemma-3-27B** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Gemma-3n-E2B** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Gemma-3n-E4B** | — | Dashboard (D) | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **GLM-4-Flash** | — | — | — | — | — | — | — | FREE (D) | — | — | — |
| **GLM-4-9B-0414** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **GLM-4.5-Air** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **GLM-4.5-Flash** | — | — | — | — | — | — | — | FREE (D) | — | — | — |
| **GLM-4.7** | — | — | — | 10 RPM (M) | — | — | — | — | — | — | — |
| **GLM-4.7-Flash** | — | — | — | — | — | — | — | FREE (D) | — | — | ~609/day (S) |
| **GLM-Z1-9B-0414** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **GPT-4o** | — | — | — | — | — | — | — | — | — | Preview (D) | — |
| **GPT-4o-Mini** | — | — | — | — | — | — | — | — | — | Preview (D) | — |
| **GPT-OSS-120B** | 30 RPM (M) | — | Dashboard (D) | 30 RPM (M) | 50/1K RPD (S) | — | — | — | — | — | ~191/day (S) |
| **GPT-OSS-20B** | 30 RPM (M) | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | ~379/day (S) |
| **GPT-5** | — | — | — | — | — | — | — | — | — | 8-12/day (D) | — |
| **GPT-5-Chat** | — | — | — | — | — | — | — | — | — | 12-20/day (D) | — |
| **GPT-5-Mini** | — | — | — | — | — | — | — | — | — | 12-20/day (D) | — |
| **GPT-5-Nano** | — | — | — | — | — | — | — | — | — | 12-20/day (D) | — |
| **Grok-3** | — | — | — | — | — | — | — | — | — | 15-30/day (D) | — |
| **Grok-3-Mini** | — | — | — | — | — | — | — | — | — | 30-50/day (D) | — |
| **Hermes-3-405B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Hunyuan-MT-7B** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **LFM-2.5-1.2B-Instruct** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **LFM-2.5-1.2B-Thinking** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Llama-3.1-8B** | 30 RPM (M) | — | Dashboard (D) | 30 RPM (M) | — | — | — | — | — | — | ~208/day (S) |
| **Llama-3.2-1B** | — | — | Dashboard (D) | — | — | — | — | — | — | — | ~1.3K/day (S) |
| **Llama-3.2-3B** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | ~726/day (S) |
| **Llama-3.3-70B** | 30 RPM (M) | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Llama-4-Scout-17B** | 30 RPM (M) | — | Dashboard (D) | — | — | — | — | — | — | — | ~210/day (S) |
| **Mistral-Large** | — | — | Dashboard (D) | — | — | Dashboard (D) | — | — | — | — | — |
| **Mistral-Small-3.1-24B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **MAI-DS-R1** | — | — | — | — | — | — | — | — | — | 8-12/day (D) | — |
| **Nemotron-3-Nano-30B** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Nemotron-Nano-9B-V2** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Qwen2-7B** | — | — | Dashboard (D) | — | — | — | FREE (D) | — | — | — | — |
| **Qwen2.5-7B** | — | — | Dashboard (D) | — | — | — | FREE (D) | — | — | — | — |
| **Qwen2.5-Coder-32B** | — | — | Dashboard (D) | — | — | — | — | — | — | — | — |
| **Qwen2.5-Coder-7B** | — | — | Dashboard (D) | — | — | — | FREE (D) | — | — | — | — |
| **Qwen3-4B** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Qwen3-8B** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **Qwen3-32B** | 60 RPM (M) | — | — | 30 RPM (M) | — | — | — | — | — | — | — |
| **Qwen3-Coder** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Qwen3-Next-80B** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Qwen3.5-4B** | — | — | — | — | — | — | FREE (D) | — | — | — | — |
| **Qwen-3-235B** | — | — | — | 30 RPM (M) | — | — | — | — | — | — | — |
| **Step-3.5-Flash** | — | — | Dashboard (D) | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Trinity-Large-Preview** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |
| **Trinity-Mini** | — | — | — | — | 50/1K RPD (S) | — | — | — | — | — | — |

**Notes:**
- **Groq**: per-model limits are explicitly published. `Llama-3.1-8B` has 7,000 RPD on the free plan; most other free chat models are closer to 1,000 RPD.
- **Gemini**: all free-tier limits are per-project and dashboard-driven. The matrix only marks models that Google still documents as free.
- **NVIDIA NIM**: public docs do not publish a static numeric rate-limit table. NVIDIA says effective limits vary by model and concurrent load, so the matrix treats NIM availability as dashboard-only.
- **Cerebras**: RPM/TPM are explicit and per-model. The current public free-tier table lists `llama3.1-8b`, `gpt-oss-120b`, `qwen-3-235b-a22b-instruct-2507`, and `zai-glm-4.7`.
- **OpenRouter**: free accounts get `50 RPD / 20 RPM`; accounts with at least $10 in credits get `1K RPD / 20 RPM` on `:free` models. That quota is shared across all free models, including eligible multimodal chat models.
- **Mistral**: a free API tier exists, but live limits remain dashboard-only.
- **Silicon Flow**: the official pricing page confirms which models are free, but does not publish RPM/TPM. The matrix therefore uses `FREE (D)` instead of unsupported numeric limits.
- **Zhipu**: the listed flash models are marked free; exact live limits are dashboard-driven, so the matrix uses `FREE (D)`.
- **Cohere**: trial/evaluation keys have a 1,000-calls/month cap; chat models run at 20 RPM per model.
- **GitHub**: GitHub publishes plan-based quotas for model tiers and named model families, but not a single per-model matrix for every Marketplace entry. `GPT-4o` / `GPT-4o-mini` are available in preview, while GPT-5, DeepSeek, and Grok families have explicit published free-tier limits.
- **Cloudflare**: daily estimates assume `1K input + 300 output tokens` and divide the free 10,000-neurons/day pool by each model's published neuron cost.

---

## Provider Deep Dives

### 1. Groq

**Why #1**: Fastest inference (560+ t/s), unlimited free tier, 30 RPM, highest reliability.

| Attribute | Value |
|-----------|-------|
| **Console** | https://console.groq.com |
| **API Keys** | https://console.groq.com/keys |
| **Models** | https://console.groq.com/docs/models |
| **Rate Limits** | https://console.groq.com/docs/rate-limits |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/groq |
| **Credit Card** | ❌ Not required |
| **Free Quota** | Unlimited (rate-limited) |
| **Speed** | 560-1,000 tokens/sec |

#### Free Chat Models (Per-Model Limits)

| Model | RPM | RPD | TPM | TPD | Context |
|-------|-----|-----|-----|-----|---------|
| `llama-3.1-8b-instant` | 30 | 7,000 | 6,000 | 500,000 | 131K |
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 | 131K |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 1,000 | 30,000 | 500,000 | 131K |
| `openai/gpt-oss-20b` | 30 | 1,000 | 8,000 | 200,000 | 131K |
| `openai/gpt-oss-120b` | 30 | 1,000 | 8,000 | 200,000 | 131K |
| `qwen/qwen3-32b` | 60 | 1,000 | 6,000 | 500,000 | 131K |

#### LiteLLM Configuration

```python
os.environ['GROQ_API_KEY'] = "gsk_..."

# Fastest option
response = completion(
    model="groq/llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Highest free-plan RPD
response = completion(
    model="groq/llama-3.1-8b-instant",  # 7,000 RPD on free plan
    messages=[...]
)
```

---

### 2. Google Gemini

**Why #2**: long context, strong reasoning, and a public free tier for Gemini Developer API experimentation.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://aistudio.google.com |
| **API Key** | https://aistudio.google.com/apikey |
| **Pricing** | https://ai.google.dev/pricing |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/gemini |
| **Credit Card** | ❌ Not required |
| **Free Quota** | Unlimited (rate-limited) |

#### Free Tier Availability

> **Dashboard-only quotas**: Google publishes free-tier availability, but not fixed public RPM/RPD/TPM tables for these models. Official rate-limit docs say limits depend on multiple factors and should be checked in [Google AI Studio](https://aistudio.google.com).

| Model / Family | Free Tier | Public quota table | Notes |
|----------------|-----------|--------------------|-------|
| `gemini-2.5-pro` | Yes | Dashboard only (D) | State-of-the-art text/reasoning model; pricing page lists free access |
| `gemini-2.5-flash` | Yes | Dashboard only (D) | 1M-context flash model with free availability |
| `gemini-2.5-flash-lite` | Yes | Dashboard only (D) | Lower-cost flash variant with free availability |
| `gemini-3-flash-preview` | Yes | Dashboard only (D) | Free preview model |
| `gemini-3.1-flash-lite-preview` | Yes | Dashboard only (D) | Free preview model |
| `gemma-3` | Yes | Dashboard only (D) | Free family on Gemini Developer API |
| `gemma-3n` | Yes | Dashboard only (D) | Free family on Gemini Developer API |

**Paid-only / not free on Developer API**: `gemini-3-pro-preview`, `gemini-3.1-pro-preview`, `imagen-4`, `veo-3`, `gemini-2.5-computer-use-preview`

**Sources**: https://ai.google.dev/pricing, https://ai.google.dev/gemini-api/docs/rate-limits

---

### 3. NVIDIA NIM

**Why #3**: 200+ catalog models, free serverless APIs for development, and very broad text-model coverage through Developer Program access.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://build.nvidia.com |
| **Models** | https://build.nvidia.com/models?filters=nimType%3Anim_type_preview&pageSize=200 |
| **API Docs** | https://docs.api.nvidia.com |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/nvidia_nim |
| **Credit Card** | ❌ Not required |
| **Free Quota** | Rate-limited for Developer Program members |

> **Official policy**: NVIDIA documents this as free access for prototyping/development through Developer Program membership. Access lasts for the duration of membership. Public FAQ/docs do not publish a fixed numeric rate-limit table; they say effective request rate varies by model and concurrent load and should be checked in your account.

#### Current Text-Capable Model Coverage

> **Text-model quota**: NVIDIA does not publish a static public per-model rate-limit table. Treat limits as dashboard-only unless you have verified live account values yourself.

| Publisher | Current visible text-capable models on the API-endpoint filtered catalog |
|-----------|---------------------------------------------------------------|
| **Meta** | `meta/llama2-70b`, `meta/llama3-8b`, `meta/llama3-70b`, `meta/llama-3.1-8b-instruct`, `meta/llama-3.1-70b-instruct`, `meta/llama-3.1-405b-instruct`, `meta/llama-3.2-1b-instruct`, `meta/llama-3.2-3b-instruct`, `meta/llama-3.3-70b-instruct`, `meta/llama-4-maverick-17b-128e-instruct`, `meta/llama-4-scout-17b-16e-instruct`, `meta/codellama-70b`, plus guard / vision variants |
| **NVIDIA** | `nvidia/nemotron-3-nano-30b-a3b`, `nvidia/nvidia-nemotron-nano-9b-v2`, `nvidia/nemotron-mini-4b-instruct`, `nvidia/llama-3.1-nemotron-nano-8b-v1`, `nvidia/llama-3.3-nemotron-super-49b-v1`, `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `nvidia/llama3-chatqa-1.5-8b`, `nvidia/usdcode`, safety / reward / translation variants |
| **Mistral AI** | `mistralai/mistral-7b-instruct`, `mistralai/mistral-7b-instruct-v0.3`, `mistralai/mixtral-8x7b-instruct`, `mistralai/mixtral-8x22b-instruct`, `mistralai/mistral-large`, `mistralai/mistral-large-3-675b-instruct-2512`, `mistralai/mistral-small-24b-instruct`, `mistralai/mistral-small-3_1-24b-instruct-2503`, `mistralai/codestral-22b-instruct-v0.1`, `mistralai/devstral-2-123b-instruct-2512`, `mistralai/mistral-nemotron`, `mistralai/magistral-small-2506` |
| **Microsoft** | `microsoft/phi-3-mini-4k-instruct`, `microsoft/phi-3-mini-128k-instruct`, `microsoft/phi-3-small-8k-instruct`, `microsoft/phi-3-small-128k-instruct`, `microsoft/phi-3-medium-4k-instruct`, `microsoft/phi-3-medium-128k-instruct`, `microsoft/phi-3.5-mini`, `microsoft/phi-4-mini-instruct`, `microsoft/phi-4-mini-flash-reasoning` |
| **Google** | `google/gemma-7b`, `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-2-27b-it`, `google/gemma-3-1b-it`, `google/gemma-3-27b-it`, `google/gemma-3n-e2b-it`, `google/gemma-3n-e4b-it`, `google/codegemma-7b`, `google/codegemma-1.1-7b`, `google/shieldgemma-9b` |
| **Qwen / DeepSeek / Others** | `qwen/qwen2-7b-instruct`, `qwen/qwen2.5-7b-instruct`, `qwen/qwen2.5-coder-7b-instruct`, `qwen/qwen2.5-coder-32b-instruct`, `qwen/qwen3-coder-480b-a35b-instruct`, `qwen/qwen3-next-80b-a3b-instruct`, `qwen/qwen3.5-122b-a10b`, `qwen/qwen3.5-397b-a17b`, `deepseek-ai/deepseek-v3.1`, `deepseek-ai/deepseek-v3.2`, `deepseek-ai/deepseek-r1-distill-qwen-7b`, `moonshotai/kimi-k2.5`, `z-ai/glm4.7`, `z-ai/glm5`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `stepfun-ai/step-3.5-flash`, `minimaxai/minimax-m2.5` |

**Catalog snapshot**: 94 API-endpoint models in the filtered preview catalog.

**Sources**: `https://build.nvidia.com`, `https://build.nvidia.com/models?filters=nimType%3Anim_type_preview&pageSize=200`, `https://docs.api.nvidia.com/nim/docs/product`, `https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317`

#### LiteLLM Configuration

```python
os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

response = completion(
    model="nvidia_nim/meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### 4. Cerebras

**Why #4**: explicit public free-tier limits, 1M tokens/day, and a small but clearly documented free model set.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://cloud.cerebras.ai |
| **Pricing** | https://www.cerebras.ai/pricing |
| **API Docs** | https://inference-docs.cerebras.ai/support/rate-limits |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/cerebras |
| **Credit Card** | ❌ Not required |
| **Free Quota** | 1M tokens/day |

#### Free Tier Models

> **Current public free-tier catalog**: Cerebras currently documents four free chat models. It also notes temporarily reduced free-tier limits for `zai-glm-4.7` and `qwen-3-235b-a22b-instruct-2507` because of demand.

| Model | RPM | TPM | Context (Free) | Scope |
|-------|-----|-----|----------------|-------|
| `llama3.1-8b` | 30 | 60,000 | 8K | Model (M) |
| `gpt-oss-120b` | 30 | 64,000 | 65K | Model (M) |
| `qwen-3-235b-a22b-instruct-2507` | 30 | 60,000 | 8K | Model (M) |
| `zai-glm-4.7` | 10 | 60,000 | 8K | Model (M) |

**Note**: Free-tier context varies by model. `gpt-oss-120b` has 65K context on free tier.

**Sources**: https://inference-docs.cerebras.ai/support/rate-limits, https://inference-docs.cerebras.ai/models/overview

#### LiteLLM Configuration

```python
os.environ['CEREBRAS_API_KEY'] = "..."

response = completion(
    model="cerebras/llama3.1-8b",
    messages=[...]
)
```

---

### 5. OpenRouter

**Why #5**: unified API, current free general-chat roster, and easy access to large open models without separate provider accounts.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://openrouter.ai |
| **Free Model Roster** | https://openrouter.ai/models?fmt=cards&max_price=0&output_modalities=text&input_modalities=text |
| **Pricing** | https://openrouter.ai/pricing |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/openrouter |
| **Credit Card** | ❌ Not required (free tier) |
| **Free Quota** | Shared across all `:free` models |

#### Free Tier Limits

> **Shared quota**: All `:free` models share a single account-wide quota.

| Condition | RPD | RPM | Scope |
|-----------|-----|-----|-------|
| No payment | 50 | 20 | Shared (S) |
| $10+ credits purchased | 1,000 | 20 | Shared (S) |

#### Current Free General-Purpose Chat Models

These are the current free general-purpose chat models derived from the official OpenRouter free-model roster and verified against `https://openrouter.ai/api/v1/models`. At the time of validation, OpenRouter exposed `24` free models total; `23` qualify for this document once general multimodal chat models are allowed, while `nvidia/nemotron-nano-12b-v2-vl:free` remains excluded as a vision-specialized model.

| Model ID | Context |
|----------|---------|
| `arcee-ai/trinity-large-preview:free` | 131K |
| `arcee-ai/trinity-mini:free` | 131K |
| `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | 32K |
| `google/gemma-3-4b-it:free` | 32K |
| `google/gemma-3-12b-it:free` | 32K |
| `google/gemma-3-27b-it:free` | 32K |
| `google/gemma-3n-e2b-it:free` | 8K |
| `google/gemma-3n-e4b-it:free` | 8K |
| `liquid/lfm-2.5-1.2b-instruct:free` | 32K |
| `liquid/lfm-2.5-1.2b-thinking:free` | 32K |
| `meta-llama/llama-3.2-3b-instruct:free` | 131K |
| `meta-llama/llama-3.3-70b-instruct:free` | 128K |
| `nousresearch/hermes-3-llama-3.1-405b:free` | 131K |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 256K |
| `nvidia/nemotron-nano-9b-v2:free` | 128K |
| `openai/gpt-oss-120b:free` | 131K |
| `openai/gpt-oss-20b:free` | 131K |
| `mistralai/mistral-small-3.1-24b-instruct:free` | 128K |
| `qwen/qwen3-4b:free` | 40K |
| `qwen/qwen3-coder:free` | 262K |
| `qwen/qwen3-next-80b-a3b-instruct:free` | 262K |
| `stepfun/step-3.5-flash:free` | 256K |
| `z-ai/glm-4.5-air:free` | 131K |

**Still excluded**: `nvidia/nemotron-nano-12b-v2-vl:free` remains out because it is positioned as a vision/document-intelligence model rather than a general-purpose chat model. `qwen/qwen3-vl-30b-a3b-thinking`, `qwen/qwen3-vl-235b-a22b-thinking`, and `openrouter/free` are not used as free general-chat matrix entries here.

**Sources**: `https://openrouter.ai/models?fmt=cards&max_price=0&output_modalities=text&input_modalities=text`, `https://openrouter.ai/api/v1/models`, `https://openrouter.ai/pricing`

### 6. Mistral

**Why #6**: Free API tier available; exact limits visible in AI Studio.

| Attribute | Value |
|-----------|-------|
| **Console** | https://console.mistral.ai |
| **Chat** | https://chat.mistral.ai |
| **Pricing** | https://mistral.ai/pricing |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/mistral |
| **Credit Card** | ❌ Not required |
| **Free Quota** | Dashboard-only (see AI Studio limits page) |

#### Free Tier Limits

> **Dashboard-only**: Mistral offers a free API tier. Exact live limits are viewed in the AI Studio limits page. Do not rely on hard-coded values.

| Metric | Value | Scope |
|--------|-------|-------|
| Rate limits | Dashboard | Account (D) |
| Available models | Check console | — |

**Source**: https://docs.mistral.ai/deployment/ai-studio/tier, https://mistral.ai/pricing

### 7. Silicon Flow (硅基流动)

**Why #7**: broad general-purpose chat-model coverage with a large set of models marked free on the official pricing page.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://siliconflow.cn |
| **Pricing** | https://siliconflow.cn/pricing |
| **Docs** | https://docs.siliconflow.cn |
|| **LiteLLM** | OpenAI-compatible endpoint |
| **Credit Card** | ❌ Not required |
| **Free Quota** | 12 general-purpose chat models explicitly marked FREE on the pricing page |

> **Note**: Silicon Flow is not a native LiteLLM provider. Use OpenAI SDK with `api_base="https://api.siliconflow.cn/v1/"` and your API key. Works with LiteLLM's OpenAI-compatible routing.

#### Free General-Purpose Chat Models

> **Important**: The official pricing page confirms which models are free, but it does **not** publish RPM/TPM tables. This section therefore lists only source-backed free general-purpose chat models and treats live limits as dashboard-only.

| Model | Free status |
|-------|-------------|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | FREE |
| `THUDM/glm-4-9b-chat` | FREE |
| `Qwen/Qwen2-7B-Instruct` | FREE |
| `Qwen/Qwen3.5-4B` | FREE |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | FREE |
| `Qwen/Qwen3-8B` | FREE |
| `tencent/Hunyuan-MT-7B` | FREE |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | FREE |
| `THUDM/GLM-Z1-9B-0414` | FREE |
| `Qwen/Qwen2.5-7B-Instruct` | FREE |
| `THUDM/GLM-4-9B-0414` | FREE |
| `internlm/internlm2_5-7b-chat` | FREE |

**Specialized exclusions from the same FREE list**: `PaddleOCR-VL-1.5`, `GLM-4.1V-9B-Thinking`, `PaddleOCR-VL`, and `DeepSeek-OCR` are excluded here because they are OCR- or vision-specialized models rather than general-purpose chat models.

**Source**: https://siliconflow.cn/pricing


---

### 8. Zhipu AI (Z.ai / GLM)

**Why #8**: strong Chinese-language support with a small public set of free flash chat models.

| Attribute | Value |
|-----------|-------|
| **Platform** | https://open.bigmodel.cn |
| **Pricing** | https://docs.z.ai/guides/overview/pricing |
| **API Docs** | https://docs.z.ai |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/zai |
| **Credit Card** | ❌ Not required |
| **Free Quota** | 3 general-purpose chat models FREE |

#### Completely Free General-Purpose Chat Models

> **Verified free general-purpose chat models**: `GLM-4-Flash`, `GLM-4.5-Flash`, and `GLM-4.7-Flash` are marked free on the official pricing page. Public rate-limit numbers are not clearly published, so treat them as dashboard-only.

| Model | Notes | Scope |
|-------|-------|-------|
| `glm-4-flash` | Older free flash chat model | Dashboard (D) |
| `glm-4.5-flash` | Current free flash chat model | Dashboard (D) |
| `glm-4.7-flash` | Newer free flash chat model | Dashboard (D) |

**Excluded specialized multimodal model**: `GLM-4.6V-Flash` is also free on the pricing page, but Z.ai documents it as a vision-language model for visual reasoning and multimodal tool use, so it remains outside this general-purpose chat matrix.

**Source**: https://docs.z.ai/guides/overview/pricing

---

### 9. Cohere

**Why #9**: Enterprise-quality models, trial/evaluation tier available.

| Attribute | Value |
|-----------|-------|
| **Dashboard** | https://dashboard.cohere.com |
| **Signup** | https://dashboard.cohere.com/welcome/register |
| **Pricing** | https://cohere.com/pricing |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/cohere |
| **Credit Card** | ❌ Not required |
| **Free Quota** | 1,000 API calls/month (trial keys) |

#### Free Tier Limits

> **Trial/evaluation keys**: Limited to 1,000 API calls/month. Chat API trial limits are 20 req/min per model. Other endpoints have separate minute limits.

| Endpoint | RPM | Monthly | Scope |
|----------|-----|---------|-------|
| Chat | 20 | 1,000 calls | Per-model (M) |
| Embed | 2,000/min | Shared | Endpoint |
| Rerank | 10 | Shared | Endpoint |

**Source**: https://cohere.com/pricing, https://docs.cohere.com


---

### 10. GitHub Models

**Why #10**: Integrated with GitHub, with free preview access across OpenAI, DeepSeek, and xAI model families.

| Attribute | Value |
|-----------|-------|
| **Marketplace** | https://github.com/marketplace/models |
| **Docs** | https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits |
| **LiteLLM** | https://docs.litellm.ai/docs/providers/github |
| **Credit Card** | ❌ Not required |
| **Free Quota** | Preview (rate limits depend on plan + model tier) |

#### Free Tier Limits

> **Official rate limits**: free playground/API usage is preview, but GitHub now publishes concrete limits by plan and model tier.

**Representative general-purpose chat models in free preview**: `GPT-4o`, `GPT-4o-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat`, `DeepSeek-R1`, `DeepSeek-R1-0528`, `MAI-DS-R1`, `xAI Grok-3`, `xAI Grok-3-Mini`

| Tier | Copilot Free | Copilot Pro | Copilot Business | Copilot Enterprise |
|------|--------------|-------------|------------------|--------------------|
| Low models | 150/day, 15 RPM | 150/day, 15 RPM | 300/day, 15 RPM | 450/day, 20 RPM |
| High models | 50/day, 10 RPM | 50/day, 10 RPM | 100/day, 10 RPM | 150/day, 15 RPM |
| Embedding models | 150/day, 15 RPM | 150/day, 15 RPM | 300/day, 15 RPM | 450/day, 20 RPM |
| `o1-preview` | n/a | 8/day, 1 RPM | 10/day, 2 RPM | 12/day, 2 RPM |
| `o1`, `o3`, `gpt-5` | n/a | 8/day, 1 RPM | 10/day, 2 RPM | 12/day, 2 RPM |
| `o1-mini`, `o3-mini`, `o4-mini`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat` | n/a | 12/day, 2 RPM | 15/day, 3 RPM | 20/day, 3 RPM |
| `DeepSeek-R1`, `DeepSeek-R1-0528`, `MAI-DS-R1` | 8/day, 1 RPM | 8/day, 1 RPM | 10/day, 2 RPM | 12/day, 2 RPM |
| `xAI Grok-3` | 15/day, 1 RPM | 15/day, 1 RPM | 20/day, 2 RPM | 30/day, 2 RPM |
| `xAI Grok-3-Mini` | 30/day, 2 RPM | 30/day, 2 RPM | 40/day, 3 RPM | 50/day, 3 RPM |

**Source**: `https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits`, `https://docs.litellm.ai/docs/providers/github`


---

### 11. Cloudflare Workers AI

**Why #11**: Edge deployment, 10K neurons/day free, good for serverless.

| Attribute | Value |
|-----------|-------|
| **Dashboard** | https://dash.cloudflare.com |
| **Docs** | https://developers.cloudflare.com/workers-ai/ |
| **Models** | https://developers.cloudflare.com/workers-ai/models/ |
| **LiteLLM** | OpenAI-compatible endpoint |
| **Credit Card** | ❌ Not required |
| **Free Quota** | 10,000 neurons/day |

#### Free Tier Limits

> **Task-based defaults**: Workers AI includes 10,000 neurons/day free allocation. Task-type RPM defaults apply with some per-model overrides.

| Task | RPM | Scope |
|------|-----|-------|
| Text Generation | 300 | Shared (S) |
| Text Embeddings | 1,500-3,000 | Shared (S) |
| Image Gen | 720 | Shared (S) |
| Speech Recognition | 720 | Shared (S) |

#### Neuron Budget Examples

> Assumption for the examples below: `1K input tokens + 300 output tokens` per request. Estimated RPD = `10,000 free neurons / neurons per request`.

| Model | Neurons / request | Est. RPD from free budget |
|-------|-------------------|---------------------------|
| `@cf/meta/llama-3.2-1b-instruct` | ~7.93 | ~1,261/day |
| `@cf/meta/llama-3.2-3b-instruct` | ~13.77 | ~726/day |
| `@cf/meta/llama-3.1-8b-instruct` | ~48.15 | ~208/day |
| `@cf/meta/llama-4-scout-17b-16e-instruct` | ~47.73 | ~210/day |
| `@cf/openai/gpt-oss-20b` | ~26.36 | ~379/day |
| `@cf/openai/gpt-oss-120b` | ~52.27 | ~191/day |
| `@cf/mistral/mistral-7b-instruct-v0.1` | ~15.19 | ~658/day |
| `@cf/zai-org/glm-4.7-flash` | ~16.42 | ~609/day |

**Sources**: https://developers.cloudflare.com/workers-ai/platform/limits/, https://developers.cloudflare.com/workers-ai/platform/pricing/


---

## Quick Start: LiteLLM Configuration

### Minimal config.yaml
```yaml
model_list:
  # Top 3 providers (unlimited)
  - model_name: groq-fast
    litellm_params:
      model: groq/llama-3.3-70b-versatile
      api_key: os.environ/GROQ_API_KEY
  
  - model_name: gemini-long
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
  
  - model_name: nvidia-nim
    litellm_params:
      model: nvidia_nim/meta/llama-3.1-70b-instruct
      api_key: os.environ/NVIDIA_NIM_API_KEY
  
  - model_name: cerebras
  # High variety
  - model_name: openrouter-free
    litellm_params:
      model: openrouter/qwen/qwen3-coder:free
      api_key: os.environ/OPENROUTER_API_KEY
  
  # All Mistral models
  - model_name: mistral
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY
  
  # Free Chinese models
  - model_name: glm-free
    litellm_params:
      model: zai/glm-4.5-flash
      api_key: os.environ/ZAI_API_KEY
  
  - model_name: silicon-free
    litellm_params:
      model: openai/Qwen/Qwen2.5-7B-Instruct
      api_key: os.environ/SILICONFLOW_API_KEY
      api_base: https://api.siliconflow.cn/v1/
```

### Environment Variables (.env)
```bash
# Top 4 (unlimited free)
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
NVIDIA_NIM_API_KEY=nvapi-...
CEREBRAS_API_KEY=...
# Model variety (50-1K RPD)
OPENROUTER_API_KEY=sk-or-...

# Dashboard limits (check console)
MISTRAL_API_KEY=...

# Chinese models (completely free)
ZAI_API_KEY=...
SILICONFLOW_API_KEY=...

# Enterprise (1K/month)
COHERE_API_KEY=...

# Developer tools
GITHUB_TOKEN=ghp_...
CLOUDFLARE_API_TOKEN=...
```

---

## Sources

All data verified from official documentation:

| Provider | Official Source |
|----------|-----------------|
| Groq | https://console.groq.com/docs/rate-limits |
| Google Gemini | https://ai.google.dev/pricing, https://ai.google.dev/gemini-api/docs/rate-limits |
| NVIDIA NIM | https://build.nvidia.com, https://build.nvidia.com/models, https://build.nvidia.com/search?label=Text-to-text, https://docs.api.nvidia.com/nim/docs/product |
| Cerebras | https://inference-docs.cerebras.ai/support/rate-limits |
| OpenRouter | https://openrouter.ai/models?fmt=cards&max_price=0&output_modalities=text&input_modalities=text, https://openrouter.ai/api/v1/models, https://openrouter.ai/pricing |
| Mistral | https://docs.mistral.ai/deployment/ai-studio/tier |
| Silicon Flow | https://siliconflow.cn/pricing |
| Zhipu AI | https://docs.z.ai/guides/overview/pricing |
| Cohere | https://docs.cohere.com/docs/rate-limits, https://cohere.com/pricing |
| GitHub Models | https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits |
| Cloudflare | https://developers.cloudflare.com/workers-ai/platform/limits/, https://developers.cloudflare.com/workers-ai/platform/pricing/ |
| LiteLLM | https://docs.litellm.ai/docs/providers |

---

## Excluded Providers

These providers were researched but **do not qualify** for inclusion:

| Provider | Reason | Details |
|----------|--------|---------|
| **SambaNova** | Time-limited | $5 credits expiring in 3 months |
| **Fireworks AI** | One-time credits | $1 starter credits only |
| **Together AI** | No free tier | Requires $5 minimum purchase |
| **Replicate** | No free tier | Pay-per-use only |
| **Perplexity** | Web-only free tier | Free tier is web interface; API requires paid plan |
| **Hugging Face Inference** | Too limited | $0.10/month credits—practically unusable |
| **Scaleway** | Unclear perpetual | €100 credit for Business accounts; unclear if recurring |
| **OVH AI Endpoints** | Unclear terms | "Discover for free" marketing language without clear perpetual terms |

---

*Last verified: March 2026. Free tier offerings change frequently—always verify with official documentation before production use.*
