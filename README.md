# MathBench: Evaluating the Theory and Application Proficiency of LLMs with a Hierarchical Mathematics Benchmark

<div align="center">

<img src="https://github.com/open-compass/opencompass/assets/28834990/c285f051-f6cb-4425-8045-863bb94095ed" width="400">
  <div> </div>
    <!-- <b><font size="3">MathBench</font></b> -->
    <div> 
  </div>
</div>

<!-- [🏰[Project Page](https://github.com/open-compass/MathBench/)]
[📚[LeaderBoard](https://github.com/open-compass/MathBench/index.html)] -->

<div align="center">

[🏰[Project Page](https://github.com/open-compass/MathBench/)]
[📚[LeaderBoard](https://open-compass.github.io/MathBench/)]

</div>

## ☀️Introduction

MathBench is an `All in One` math dataset for language model evaluation, with: 
- **A Sophisticated Five-Stage Difficulty Mechanism:** Unlike the usual mathematical datasets that can only evaluate a single difficulty level or have a mix of unclear difficulty levels, MathBench provides 3709 questions with a gradient difficulty division by education stages, ranging from basic arithmetic to primary, middle, high school, and college levels, allowing you to get a clear overview of the comprehensive difficulty evaluation results.
- **Bilingual Gradient Evaluation:** Apart from the basic calculation part which is language-independent, MathBench provides questions in both Chinese and English for the four-stage difficulty datasets from primary to college.
- **Implementation of the Robust Circular Evaluation (CE) Method:** MathBench use CE as the main evaluation method for questions. Compared to the general Accuracy evaluation method, CE requires the model to answer the same multiple-choice question multiple times, with the order of the options changing each time. The model is considered correct on this question only if all answers are correct. The results of CE can reflect the model's capabilities more realistically, providing more valuable evaluation results.
- **Support for Basic Theory Questions:** For every stage, MathBench provides questions that cover the basic theory knowledge points of the corresponding stage, to ascertain whether the model has genuinely mastered the fundamental concepts of each stage or merely memorized the answers.

<!-- CE utilizes a circular evaluation mechanism to mitigate the model's biased tendencies, such as consistently favoring option A or yielding entirely different results across multiple responses. During the evaluation of a multiple-choice question, CE performs several assessments. After each question-answer interaction, the order of the options is rearranged through a "circular" mechanism (for instance, ABCD becomes BCDA). A question is only deemed correct if all responses across these evaluations are accurate. Within MathBench, we employ CE-4, meaning each question undergoes four rounds of evaluation. -->

## 🚀 What's New
- **[2024.3.14]** We release the complete version of MathBench, which includes a total of 3,709 problems in both Chinese and English. The dataset covers both **applied and theoretical** math problems. Each problem is labeled with a **three-level taxonomy**. 🎉🎉🎉
- **[2024.1.26]** We release the **Application questions** of MathBench. 🎉🎉🎉

## 🌲Dataset Structure
<div align="center">
 <img src="https://github.com/hpcaitech/ColossalAI/assets/28834990/866e88d6-4d4f-4e19-aadb-bcb047fffe76" width="800"/>
</div>


## 📒Model Performance
We use zero-shot CoT set for multiple-choice questions and few-shot (8) CoT set for all textual questions. The results are shown in the following table, we present the results with common Accuracy and Circular Evaluation (CE) metrics. 

Here is the overall average result of **MathBench**:


| Models                      | ACC Average | CE Average |
|-----------------------------|-------------|------------|
| **API Models**              |             |            |
| GPT-3.5                     | 66.1        | 48.1       |
| Qwen-Max                    | 75.2        | 62.9       |
| GLM4                        | 73.0        | 60.3       |
| Claude-3-Opus               | 77.4        | 64.2       |
| GPT-4                       | **78.8**    | **68.9**   |
| **Open-Source Chat Models** |             |            |
| ChatGLM3-6B                 | 43.7        | 24.3       |
| Yi-6B-Chat                  | 42.5        | 23.5       |
| InternLM2-Chat-7B           | 59.3        | 42.7       |
| Qwen-7B-Chat                | 51.8        | 34.3       |
| Deepseek-7B-Chat            | 43.5        | 23.9       |
| Baichuan2-13B-Chat          | 49.5        | 32.1       |
| Qwen-14B-Chat               | 64.1        | 44.4       |
| InternLM2-Chat-20B          | **64.7**    | **47.4**   |
| Yi-34B-Chat                 | 54.9        | 39.0       |
| Deepseek-67B-Chat           | 65.3        | 52.0       |
| Qwen-72B-Chat               | **72.2**       | **61.1**       |
| **Mathematical Models**     |             |            |
| MammoTH-7B                  | 28.5        | 11.7       |
| Metamath-Llemma-7B          | 39.4        | 25.1       |
| InternLM2-Chat-Math-7B      | 64.0        | 46.0       |
| Deepseek-Math-7B-Instruct   | 63.9        | 48.3       |
| Deepseek-Math-7B-RL         | **69.6**    | **58.7**   |
| MammoTH-13B                 | 38.9        | 20.2       |
| InternLM2-Chat-Math-20B     | 64.6        | 51.8       |
| MammoTH-70B                 | 49.9        | 34.2       |



Here is the CE result of **MathBench** with 5-level difficulty divisions🪜:


| Models                      | Arithmetic | Primary | Middle | High  | College |
|-----------------------------|------------|---------|--------|-------|---------|
| **API Models**              |            |         |        |       |         |
| GPT-3.5                     | 70.3       | 67.9    | 39.3   | 30.6  | 32.2    |
| Qwen-Max                    | 69.3       | 84.2    | 62.7   | 51.4  | 47.0    |
| GLM4                        | 61.3       | 83.0    | 64.0   | 52.1  | 41.2    |
| Claude-3-Opus               | **84.7**   | **86.1**| 63.5   | 48.2  | 38.7    |
| GPT-4                       | 76.3       | 82.9    | **69.8**| **56.6**| **59.0**|
| **Open-Source Chat Models** |            |         |        |       |         |
| ChatGLM3-6B                 | 41.0       | 40.5    | 21.4   | 11.5  | 6.3     |
| Yi-6B-Chat                  | 35.7       | 41.1    | 20.3   | 11.5  | 9.1     |
| InternLM2-Chat-7B           | 53.0       | 67.5    | 41.0   | 29.6  | 22.6    |
| Qwen-7B-Chat                | 51.3       | 50.2    | 32.6   | 20.2  | 17.3    |
| Deepseek-7B-Chat            | 46.0       | 39.3    | 15.5   | 9.6   | 9.2     |
| Baichuan2-13B-Chat          | 46.0       | 54.2    | 29.5   | 16.6  | 14.3    |
| Qwen-14B-Chat               | 64.7       | 66.1    | 49.2   | 32.8  | 27.2    |
| InternLM2-Chat-20B          | 62.7       | 70.0    | 47.4   | 33.7  | 23.3    |
| Yi-34B-Chat                 | 51.0       | 64.8    | 38.0   | 23.2  | 17.8    |
| Deepseek-67B-Chat           | 61.3       | 77.2    | 48.4   | 36.3  | 36.8    |
| Qwen-72B-Chat               | **72.0**   | **80.1**   | **64.8**| **47.8**| **40.8**    |
| **Mathematical Models**     |            |         |        |       |         |
| MammoTH-7B                  | 26.7       | 18.1    | 5.3    | 4.8   | 3.7     |
| Metamath-Llemma-7B          | 48.7       | 35.3    | 16.1   | 15.5  | 10.1    |
| InternLM2-Chat-Math-7B      | 53.7       | 66.0    | 49.0   | 34.3  | 26.9    |
| Deepseek-Math-7B-Instruct   | 61.0       | 73.7    | 42.2   | 34.9  | 29.9    |
| Deepseek-Math-7B-RL         | **67.7**   | **80.8**| **57.2** | **45.4**| **42.7**|
| MammoTH-13B                 | 35.0       | 34.8    | 10.7   | 9.9   | 10.6    |
| InternLM2-Chat-Math-20B     | 58.7       | 71.1    | 55.5   | 41.8  | 31.9    |
| MammoTH-70B                 | 35.7       | 59.3    | 28.1   | 23.6  | 24.5    |



## 🔊Application Scores with Stages
 Models exhibit similar performances in Arithmetic and Primary stages, while demonstrating a clear performance decline from Primary to College stages.
<div align="center">
 <img src="https://github.com/hpcaitech/ColossalAI/assets/28834990/ad967953-2857-4e85-86d2-e1a9504ff38b" width="800"/>
</div>

## 🖋Inference MathBench with OpenCompass
[OpenCompass](https://github.com/open-compass/opencompass) is a toolkit for evaluating the performance of large language models (LLMs). There are steps for inference MathBench with OpenCompass:
1. Install OpenCompass
```
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```
2. Prepare the dataset
```
# Download dataset from release file and copy to data/ folder
mkdir data
cp -rf mathbench ./data/ 
```
3. Inference MathBench
```
# Inference MathBench with hf_llama2_7b_chat model
python run.py --models hf_llama2_7b_chat --datasets mathbench_gen
```
You can also evaluate HuggingFace models via command line. 
```
python run.py --datasets mathbench_gen \
--hf-path meta-llama/Llama-2-7b-chat-hf \  # HuggingFace model path
--model-kwargs device_map='auto' \  # Arguments for model construction
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--batch-size 8 \  # Batch size
--no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
--num-gpus 1  # Number of minimum required GPUs
--summarizer summarizers.mathbench_v1 # Summarizer for MathBench
```


# Citation and Tech Report
To be appended.