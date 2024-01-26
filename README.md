# MathBench: A Comprehensive Multi-Level Difficulty Mathematics Evaluation Dataset

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
- **A Sophisticated Five-Level Difficulty Mechanism:** Unlike the usual mathematical datasets that can only evaluate a single difficulty level or have a mix of unclear difficulty levels, MathBench provides 1500 questions with a gradient difficulty division by education stages, ranging from basic calculations to primary, middle, high school, and university levels, allowing you to get a clear overview of the comprehensive difficulty evaluation results.
- **Multi-Language Gradient Evaluation:** Apart from the basic calculation part which is language-independent, MathBench provides questions in both Chinese and English for the four-level difficulty datasets from primary to university level.
- **Implementation of the Robust Circular Evaluation (CE) Method:** MathBench use CE as the main evaluation method for questions. Compared to the general Accuracy evaluation method, CE requires the model to answer the same multiple-choice question multiple times, with the order of the options changing each time. The model is considered correct on this question only if all answers are correct. The results of CE can reflect the model's capabilities more realistically, providing more valuable evaluation results.
- **Support for Basic Knowledge Point Questions (Coming Soon):** For every difficulty level, MathBench provides questions that cover the basic knowledge points of the corresponding level, to ascertain whether the model has genuinely mastered the fundamental concepts of each level or merely memorized the answers.

<!-- CE utilizes a circular evaluation mechanism to mitigate the model's biased tendencies, such as consistently favoring option A or yielding entirely different results across multiple responses. During the evaluation of a multiple-choice question, CE performs several assessments. After each question-answer interaction, the order of the options is rearranged through a "circular" mechanism (for instance, ABCD becomes BCDA). A question is only deemed correct if all responses across these evaluations are accurate. Within MathBench, we employ CE-4, meaning each question undergoes four rounds of evaluation. -->

## 🚀 What's New

- **[2024.1.26]** We release the [data]() of MathBench. 🎉🎉🎉

## 🌲Dataset Structure
<div align="center">
 <img src="https://github.com/InternLM/InternLM-Math/assets/28834990/a5cc2887-5107-4f5a-b04b-48adf7be8349" width="600"/>
</div>


## 📒Model Performance
We use zero-shot CoT set for multiple-choice questions and few-shot (8) CoT set for all textual questions. The results are shown in the following table, we present the results with common Accuracy and Circular Evaluation (CE) metrics. 

Here is the overall average result of **MathBench**:

| Method                 | CE Average | Acc Average |
|------------------------|------------|-------------|
| Internlm2-Chat-7B      | 31.07      | 46.71       |
| Internlm2-Chat-20B     | 40.89      | 54.67       |
| Qwen-7B-Chat           | 24.78      | 39.32       |
| Qwen-14B-Chat          | 35.07      | 51.73       |
| Qwen-72B-Chat          | 45.63      | 60.40       |
| Deepseek-7B-Chat       | 17.59      | 33.25       |
| Deepseek-67B-Chat      | 40.52      | 52.64       |
| Chatglm3-6B            | 17.41      | 33.80       |
| Mammoth-7B             | 9.56       | 22.96       |
| Mammoth-13B            | 16.11      | 31.22       |
| Mammoth-70B            | 22.93      | 37.96       |
| GPT-3.5-Turbo          | 31.45      | 52.04       |
| GPT-4                  | 52.85      | 70.48       |

Here is the CE result of **MathBench** with 5-level difficulty divisions🪜:

| Model                  | Calculate | Primary | Middle | High   | College |
|------------------------|-----------|---------|--------|--------|---------|
| Internlm2-Chat-7B      | 53.00     | 67.67   | 25.00  | 14.34  | 6.34    |
| Internlm2-Chat-20B     | 62.67     | 75.67   | 39.67  | 23.67  | 13.67   |
| Qwen-7B-Chat           | 51.33     | 48.67   | 22.00  | 9.67   | 5.50    |
| Qwen-14B-Chat          | 64.67     | 61.34   | 36.67  | 19.67  | 7.84    |
| Qwen-72B-Chat          | 72.00     | 70.67   | 52.67  | 30.67  | 15.34   |
| Deepseek-7B-Chat       | 46.00     | 45.67   | 5.00   | 3.67   | 1.84    |
| Deepseek-67B-Chat      | 61.33     | 76.34   | 33.00  | 19.00  | 23.34   |
| Chatglm3-6B            | 41.00     | 42.67   | 11.67  | 3.00   | 0.50    |
| Mammoth-7B             | 26.67     | 24.33   | 3.00   | 1.34   | 1.00    |
| Mammoth-13B            | 35.00     | 41.67   | 5.00   | 4.00   | 4.34    |
| Mammoth-70B            | 35.67     | 60.67   | 11.00  | 8.34   | 5.34    |
| GPT-3.5-Turbo          | 71.00     | 66.67   | 16.67  | 14.34  | 8.34    |
| GPT-4                  | 73.00     | 89.34   | 52.34  | 33.00  | 26.67   |

<!-- <div align="center">
<img src="https://github.com/InternLM/InternLM-Math/assets/28834990/1026cd1a-199f-43ea-bb0d-4aa3fce53442" width="600">
</div> -->


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
--summarizer summarizers.mathbench # Summarizer for MathBench
```


# Citation and Tech Report
To be appended.