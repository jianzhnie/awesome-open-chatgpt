<div align="center">
    <h1>Awesome Open Chatgpt</h1>
    <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg"/></a>
</div>

ChatGPT is GPT-3.5 finetuned with RLHF (Reinforcement Learning with Human Feedback) for human instruction and chat.

Alternatives are projects featuring different instruct finetuned language models for chat. 
Projects are **not** counted if they are:
- Alternative frontend projects which simply call OpenAI's APIs. 
- Using language models which are not finetuned for human instruction or chat.

Tags:
-   Bare: only source code, no data, no model's weight, no chat system
-   Standard: yes data, yes model's weight, bare chat via API
-   Full: full yes data, yes model's weight, fancy chat system including TUI and GUI
-   Complicated: semi open source, not really open source, based on closed model, etc...

Other revelant lists:
- [yaodongC/awesome-instruction-dataset](https://github.com/yaodongC/awesome-instruction-dataset): A collection of open-source dataset to train instruction-following LLMs (ChatGPT,LLaMA,Alpaca)

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Open source ChatGPT list](#open-source-chatgpt-list)
- [The template](#the-template)
- [The list](#the-list)
  - [lucidrains/PaLM-rlhf-pytorch](#lucidrainspalm-rlhf-pytorch)
  - [LAION-AI/Open-Assistant](#laion-aiopen-assistant)
  - [ColossalAI/Chat](#colossalaichat)
  - [nebuly-ai/nebullvm](#nebuly-ainebullvm)
  - [Stability-AI/StableLM](#stability-aistablelm)
  - [DeepSpeed-Chat](#deepspeed-chat)
  - [togethercomputer/OpenChatKit](#togethercomputeropenchatkit)
  - [tatsu-lab/stanford\_alpaca](#tatsu-labstanford_alpaca)
  - [THUDM/ChatGLM-6B](#thudmchatglm-6b)
  - [databrickslabs/dolly](#databrickslabsdolly)
  - [h2oai/h2ogpt](#h2oaih2ogpt)
  - [clue-ai/ChatYuan](#clue-aichatyuan)
  - [nomic-ai/gpt4all](#nomic-aigpt4all)
  - [oobabooga/text-generation-webui](#oobaboogatext-generation-webui)
  - [KoboldAI/KoboldAI-Client](#koboldaikoboldai-client)
  - [lm-sys/FastChat](#lm-sysfastchat)
  - [young-geng/EasyLM](#young-gengeasylm)
  - [Lightning-AI/lit-llama](#lightning-ailit-llama)
  - [LianjiaTech/BELLE](#lianjiatechbelle)
  - [BlinkDL/ChatRWKV](#blinkdlchatrwkv)
  - [bigscience-workshop/xmtf](#bigscience-workshopxmtf)
  - [carperai/trlx](#carperaitrlx)
  - [LianjiaTech/BELLE](#lianjiatechbelle-1)
  - [ethanyanjiali/minChatGPT](#ethanyanjialiminchatgpt)
  - [cerebras/Cerebras-GPT](#cerebrascerebras-gpt)
  - [TavernAI/TavernAI](#tavernaitavernai)
- [Other LLaMA-derived projects:](#other-llama-derived-projects)

# Open source ChatGPT list

list of open source works to implement ChatGPT-like models.

| institution                                   | model                                                        | language | base model                              | Tuning dataset                                               |                         main feature                         |
| :-------------------------------------------- | :----------------------------------------------------------- | -------- | :-------------------------------------- | ------------------------------------------------------------ | :----------------------------------------------------------: |
| Meta                                          | [LLaMA](https://github.com/facebookresearch/llama)           | en       | -                                       | [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | LLaMA-13B outperforms GPT-3(175B) and LLaMA-65B is competitive to PaLM-540M.<br />Base model for most follow-up works. |
| @ggerganov                                    | [llama.cpp](https://github.com/ggerganov/llama.cpp)          | en       | LLaMA                                   | NA                                                           | c/cpp implement of llama and some other models, using quantization. |
| Stanford                                      | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)       | en       | LLaMA-7B                                | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | use 52K instruction-following data generated by Self-Instructt techniques to fine-utne 7B LLaMA,<br /> the resulting model,  Alpaca, behaves similarly to the `text-davinci-003` model on the Self-Instruct instruction-following evaluation suite.<br />Alpaca has inspired many follow-up models. |
| LianJia                                       | [BELLE](https://github.com/LianjiaTech/BELLE)                | en/zh    | BLOOMZ-7B1-mt                           | [1.5M中文数据集](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M) |       maybe the first Chinese model to follow Alpaca.        |
| Tsinghua                                      | [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)            | en/zh    | GLM                                     | NA                                                           | well-known Chinese model, in chat mode, and can run on single GPU. |
| Databricks                                    | [Dolly](https://github.com/databrickslabs/dolly)             | en       | GPT-J 6B                                | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | use Alpaca data to fine-tune a 2-year-old model: GPT-J, which exhibits surprisingly high quality<br /> instruction following behavior not characteristic of the foundation model on which it is based. |
| @tloen                                        | [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)          | en       | LLaMA-7B                                | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) | trained within hours on a single RTX 4090,<br />reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf),<br />and can run on a Raspberry pi. |
| ColossalAI                                    | [ColossalChat](https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/README.md) | en/zh    | LLaMA-7B                                | [InstructWild Data](https://github.com/XueFuzhao/InstructionWild/tree/main/data) | provides a unified large language model framework, including:<br />Supervised datasets collection<br />Supervised instructions fine-tuning<br />Reward model training<br />RLHF<br />Quantization inference<br />Fast model deploying<br />Perfectly integrated with the Hugging Face ecosystem |
| Shanghai AI Lab                               | [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) | en       | LLaMA-7B                                | NA                                                           | Fine-tuning LLaMA to follow instructions within 1 Hour and 1.2M Parameters |
| PhoebusSi                                     | [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)        | en/zh    | LLaMA<br />ChatGLM<br />BLOOM           |                                                              | extend CoT data to Alpaca to boost its reasoning ability.<br />aims to build an instruction finetuning (IFT) platform with extensive instruction collection (especially the CoT datasets)<br /> and a unified interface for various large language models. |
| AetherCortex                                  | [Llama-X](https://github.com/AetherCortex/Llama-X)           | en       | LLaMA                                   |                                                              |    Open Academic Research on Improving LLaMA to SOTA LLM     |
| Together                                      | [OpenChatKit](https://github.com/togethercomputer/OpenChatKit) | en       | GPT-NeoX-20B                            | [laion/OIG](https://huggingface.co/datasets/laion/OIG)       | OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications.<br /> The kit includes an instruction-tuned language models, a moderation model, and an extensible retrieval system for including <br />up-to-date responses from custom repositories. |
| nomic-ai                                      | [GPT4All](https://github.com/nomic-ai/gpt4all)               | en       | LLaMA                                   |                                                              | trained on a massive collection of clean assistant data including code, stories and dialogue |
| @ymcui                                        | [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | en/zh    | LLaMA-7B/13B                            |                                                              | expand the Chinese vocabulary based on the original LLaMA and use Chinese data for secondary pre-training,<br /> further enhancing Chinese basic semantic understanding. Additionally, the project uses Chinese instruction data<br /> for fine-tuning on the basis of the Chinese LLaMA, significantly improving the model's understanding and execution of instructions. |
| UC Berkley<br />Stanford<br />CMU             | [Vicuna](https://github.com/lm-sys/FastChat)                 | en       | LLaMA-13B                               |                                                              |          Impressing GPT-4 with 90% ChatGPT Quality           |
| @NouamaneTazi                                 | [bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp)     | en/zh    | BLOOM                                   |                                                              |           C++ implementation for BLOOM inference.            |
| HKUST                                         | [LMFlow](https://github.com/OptimalScale/LMFlow)             | en/zh    | LLaMA<br />Galatica<br />GPT-2<br />... |                                                              | An extensible, convenient, and efficient toolbox for finetuning large machine learning models, designed to be user-friendly,<br /> speedy and reliable, and accessible to the entire community. |
| [Cerebras Systems](https://www.cerebras.net/) | [Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-13B) | en       | -                                       |                                                              | Pretrained LLM, GPT-3 like, Commercially available, efficiently trained on the[Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer,<br />trained in accordance with [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) (20 tokens per model parameter) which is compute-optimal. |
| UT Southwestern/<br />UIUC/OSU/HDU            | [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)        | en       | LLaMA                                   | [ChatDoctor Dataset](https://github.com/Kent0n-Li/ChatDoctor#1-chatdoctor-dataset) |  Maybe the first domain-specific chat model tuned on LLaMA.  |
| LAION-AI                                      | [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) | en/zh    | Llama                                   | [OpenAssistant/oasst1](https://github.com/LAION-AI/Open-Assistant/blob/main/docs/docs/data/datasets.md) | Open Assistant is a project meant to give everyone access to a great chat based large language model. |
| project-baize                                 | [baize-chatbot](https://github.com/project-baize/baize-chatbot) | En/zh    | llama                                   | [baize-chatbot](https://github.com/project-baize/baize-chatbot/tree/main/data) | Baize is an open-source chat model trained with [LoRA](https://github.com/microsoft/LoRA). It uses 100k dialogs generated by letting ChatGPT chat with itself. We also use Alpaca's data to improve its performance. We have released 7B, 13B and 30B models. Please refer to the [paper](https://arxiv.org/pdf/2304.01196.pdf) for more details. |

# The template

Append the new project at the end of file

```markdown
## [{owner}/{project-name}]{https://github.com/link/to/project}

Description goes here

Tags: Bare/Standard/Full/Complicated
```

# The list

## [lucidrains/PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)

Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Basically ChatGPT but with PaLM

Tags: Bare
stars: ⭐⭐⭐

## [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant) 

OpenAssistant is a chat-based assistant that understands tasks, can interact with third-party systems, and retrieve information dynamically to do so.

Related links:

- [huggingface.co/OpenAssistant](https://huggingface.co/OpenAssistant)
- [r/OpenAssistant/](https://www.reddit.com/r/OpenAssistant/)

Tags: Full
Stars: ⭐⭐⭐⭐⭐


## [ColossalAI/Chat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)

ColossalChat implement LLM with RLHF, powered by the Colossal-AI project.

Tags: full
Stars: ⭐⭐⭐⭐⭐

## [nebuly-ai/nebullvm](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)

由于 LLaMA 大模型系列没有使用 RLHF 方法，因此初创公司 Nebuly AI 开源了 RLHF 版 LLaMA（ChatLLaMA）的训练方法。它的训练过程类似 ChatGPT，该项目允许基于预训练的 LLaMA 模型构建 ChatGPT 形式的服务。与 ChatGPT 相比，LLaMA 架构更小，但训练过程和单 GPU 推理速度更快，成本更低；该库还支持所有的 LLaMA 模型架构（7B、13B、33B、65B），因此用户可以根据训练时间和推理性能偏好对模型进行微调。

ChatLLAMA is a chatbot powered by LLaMA, a large language model finetuned with RLHF.

Tags: Full
Stars: ⭐⭐⭐⭐⭐

## [Stability-AI/StableLM](https://github.com/Stability-AI/StableLM)

This repository contains Stability AI's ongoing development of the StableLM series of language models and will be continuously updated with new checkpoints.

Related links:

- [huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat)

Tags: Full

## [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. A fast, affordable, scalable and open system framework for enabling end-to-end Reinforcement Learning Human Feedback (RLHF) training experience to generate high-quality ChatGPT-style models at all scales.

Related links:

- https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
- https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

## [togethercomputer/OpenChatKit](https://github.com/togethercomputer/OpenChatKit)

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. 

Related links:

- [spaces/togethercomputer/OpenChatKit](https://huggingface.co/spaces/togethercomputer/OpenChatKit)

Tags: Full
Stars: ⭐⭐⭐⭐⭐

## [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

This is the repo for the Stanford Alpaca project, which aims to build and share an instruction-following LLaMA model.

Tags: Complicated
Stars: ⭐⭐⭐

**Resources:** 

- Blog: [Stanford CRFM](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- GitHub: [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- Demo: [Alpaca-LoRA](https://huggingface.co/spaces/tloen/alpaca-lora) (The official demo was drop and this is a recreation of Alpaca model)

## [lm-sys/FastChat](https://github.com/lm-sys/FastChat)

An open platform for training, serving, and evaluating large language model based chatbots. Vicuna is an open-source chatbot with 13B parameters trained by fine-tuning LLaMA on user conversations data collected from ShareGPT.com, a community site users can share their ChatGPT conversations. Based on evaluations done, the model has a more than 90% quality rate comparable to OpenAI's ChatGPT and Google's Bard, which makes this model one of the top opensourced models when looking at feature parity to ChatGPT. 

**Resources:** 

- Blog post: [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://vicuna.lmsys.org/)
- GitHub: [lm-sys/FastChat](https://github.com/lm-sys/FastChat#fine-tuning)
- Demo: [FastChat (lmsys.org)](https://chat.lmsys.org/)

## [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).

Related links:

- Alternative Web UI: [Akegarasu/ChatGLM-webui](https://github.com/Akegarasu/ChatGLM-webui)
- Slim version (remove 20K image tokens to reduce memory usage): [silver/chatglm-6b-slim](https://huggingface.co/silver/chatglm-6b-slim)
- Fintune ChatGLM-6b using low-rank adaptation (LoRA): [lich99/ChatGLM-finetune-LoRA](https://github.com/lich99/ChatGLM-finetune-LoRA)
- Deploying ChatGLM on Modelz: [tensorchord/modelz-ChatGLM](https://github.com/tensorchord/modelz-ChatGLM)
- Docker image with built-on playground UI and streaming API compatible with OpenAI, using [Basaran](https://github.com/hyperonym/basaran): [peakji92/chatglm:6b](https://hub.docker.com/r/peakji92/chatglm/tags)

Tags: Full

## [databrickslabs/dolly](https://github.com/databrickslabs/dolly)

Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform. Script to fine tune [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset. Insightful if you want to fine tune LLMs.

Related links:

- [6b model card](https://huggingface.co/databricks/dolly-v1-6b)

Tags: Standard

Stars: ⭐⭐⭐⭐

## [h2oai/h2ogpt](https://github.com/h2oai/h2ogpt)

h2oGPT - The world's best open source GPT

- Open-source repository with fully permissive, commercially usable code, data and models
- Code for preparing large open-source datasets as instruction datasets for fine-tuning of large language models (LLMs), including prompt engineering
- Code for fine-tuning large language models (currently up to 20B parameters) on commodity hardware and enterprise GPU servers (single or multi node)
- Code to run a chatbot on a GPU server, with shareable end-point with Python client API
- Code to evaluate and compare the performance of fine-tuned LLMs

Related links:

- [h2oGPT 20B](https://gpt.h2o.ai/)
- [🤗 h2oGPT 12B #1](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot)
- [🤗 h2oGPT 12B #2](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)

Tags: Full

## [clue-ai/ChatYuan](https://github.com/clue-ai/ChatYuan)

ChatYuan: Large Language Model for Dialogue in Chinese and English (The repos are mostly in Chinese)

Related links:

- [A bit translated readme to English](https://github.com/nichtdax/awesome-totally-open-chatgpt/issues/18#issuecomment-1492826662)

Tags: Full

## [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)

[GPT4all](https://github.com/nomic-ai/gpt4all) is a community-driven project trained on a massive curated collection of written texts of assistant interactions, including code, stories, depictions, and multi-turn dialogue. The team has provided datasets, model weights, data curation processes, and training code to promote the open-source model. There is also a release of a quantized 4-bit version of the model that is able to run on your laptop as the memory and computation power required is less. A Python client is also available that you can use to interact with the model.

Tags: full
Stars: ⭐⭐⭐⭐⭐

## [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.

Tags: Full

## [KoboldAI/KoboldAI-Client](https://github.com/KoboldAI/KoboldAI-Client)

This is a browser-based front-end for AI-assisted writing with multiple local & remote AI models. It offers the standard array of tools, including Memory, Author’s Note, World Info, Save & Load, adjustable AI settings, formatting options, and the ability to import existing AI Dungeon adventures. You can also turn on Adventure mode and play the game like AI Dungeon Unleashed.

Tags: Full


## [young-geng/EasyLM](https://github.com/young-geng/EasyLM)
Large language models (LLMs) made easy, EasyLM is a one stop solution for pre-training, finetuning, evaluating and serving LLMs in JAX/Flax. EasyLM can scale up LLM training to hundreds of TPU/GPU accelerators by leveraging JAX's pjit functionality.


## [Lightning-AI/lit-llama](https://github.com/Lightning-AI/lit-llama) 

Implementation of the LLaMA language model based on nanoGPT.


## [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)

本项目目标是促进中文对话大模型开源社区的发展，愿景做能帮到每一个人的LLM Engine。现阶段本项目基于一些开源预训练大语言模型（如BLOOM），针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

## [BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV)

[ChatRWKV](https://github.com/BlinkDL/ChatRWKV) is an open-source chatbot powered by RWKV, an RNN with Transformer-level LLM performance language model. Model results are comparable with those of ChatGPT. The model uses RNNs. Fine-tuning of the model was done using Stanford Alpaca and other datasets.

Tags: Full

## [bigscience-workshop/xmtf](https://github.com/bigscience-workshop/xmtf)

This repository provides an overview of all components used for the creation of BLOOMZ & mT0 and xP3 introduced in the paper [Crosslingual Generalization through Multitask Finetuning](https://arxiv.org/abs/2211.01786).

Related links:
- [bigscience/bloomz](https://huggingface.co/bigscience/bloomz)
- [bigscience/mt0-base](https://huggingface.co/bigscience/mt0-base)

Tags: Standard

## [carperai/trlx](https://github.com/carperai/trlx)

 A repo for distributed training of language models with Reinforcement Learning via Human Feedback (RLHF), supporting online RL up to 20b params and offline RL to larger models. Basically what you would use to finetune GPT into ChatGPT. 

Tags: Bare

## [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)

The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models. This project optimizes Chinese performance in addition to original Stanford Alpaca. The model finetuning uses only data generated via ChatGPT (without other data). This repo contains: 175 chinese seed tasks used for generating the data, code for generating the data, 0.5M generated data used for fine-tuning the model, model finetuned from BLOOMZ-7B1-mt on data generated by this project.

Related links:
- [English readme](https://github.com/LianjiaTech/BELLE#-belle-be-large-language-model-engine-1)

Tags: Standard

## [ethanyanjiali/minChatGPT](https://github.com/ethanyanjiali/minChatGPT)

A minimum example of aligning language models with RLHF similar to ChatGPT

Related links:
- [huggingface.co/ethanyanjiali/minChatGPT](https://huggingface.co/ethanyanjiali/minChatGPT)

Tags: Standard

## [cerebras/Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-6.7B)

7 open source GPT-3 style models with parameter ranges from 111 million to 13 billion, trained using the [Chinchilla](https://arxiv.org/abs/2203.15556) formula. Model weights have been released under a permissive license (Apache 2.0 license in particular).

Related links:
- [Announcement](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
- [Models with other amount of parameters](https://huggingface.co/cerebras)

Tags: Standard

## [TavernAI/TavernAI](https://github.com/TavernAI/TavernAI)

Atmospheric adventure chat for AI language model **Pygmalion** by default and other models such as **KoboldAI**, ChatGPT, GPT-4

Tags: Full

# Other LLaMA-derived projects:

- [project-baize/baize-chatbot](https://github.com/project-baize/baize-chatbot) Baize is an open-source chat model trained with LoRA. It uses 100k dialogs generated by letting ChatGPT chat with itself. We also use Alpaca's data to improve its performance. We have released 7B, 13B and 30B models. Please refer to the paper for more details.
- [pointnetwork/point-alpaca](https://github.com/pointnetwork/point-alpaca) Released weights recreated from Stanford Alpaca, an experiment in fine-tuning LLaMA on a synthetic instruction dataset.
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) Code for rproducing the Stanford Alpaca results using low-rank adaptation (LoRA).
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) Ports for inferencing LLaMA in C/C++ running on CPUs, supports alpaca, gpt4all, etc.
- [setzer22/llama-rs](https://github.com/setzer22/llama-rs) Rust port of the llama.cpp project.
- [juncongmoo/chatllama](https://github.com/juncongmoo/chatllama) Open source implementation for LLaMA-based ChatGPT runnable in a single GPU.
- [Lightning-AI/lit-llama](https://github.com/Lightning-AI/lit-llama) Implementation of the LLaMA language model based on nanoGPT.
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) Demo, data and code to train an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMA.
- [hpcaitech/ColossalAI#ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) An open-source solution for cloning ChatGPT with a complete RLHF pipeline.
- [lm-sys/FastChat](https://github.com/lm-sys/FastChat) An open platform for training, serving, and evaluating large language model based chatbots.
- [nsarrazin/serge](https://github.com/nsarrazin/serge) A web interface for chatting with Alpaca through llama.cpp. Fully dockerized, with an easy to use API.

