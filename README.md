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

<!-- [🏰[Project Page](https://github.com/open-compass/MathBench/)] -->
[📄[Paper](https://arxiv.org/abs/2405.12209)]
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
- **[2024.5.20]** MathBench has been accepted by **ACL2024** (findings), We also released performance of more models such as GPT-4o, Qwen-Max-0428, Llama3 and DeepSeek-V2-API on MathBench. 🎉🎉🎉
- **[2024.3.14]** We release the complete version of MathBench, which includes a total of 3,709 problems in both Chinese and English. The dataset covers both **applied and theoretical** math problems. Each problem is labeled with a **three-level taxonomy**. 🎉🎉🎉
- **[2024.1.26]** We release the **Application questions** of MathBench. 🎉🎉🎉

## 🌲Dataset Structure
<div align="center">
 <img src="https://github.com/hpcaitech/ColossalAI/assets/28834990/866e88d6-4d4f-4e19-aadb-bcb047fffe76" width="800"/>
</div>


## 📒Model Performance
We use zero-shot CoT set for multiple-choice questions and few-shot (8) CoT set for all textual questions. The results are shown in the following table, we present the results with common Accuracy and Circular Evaluation (CE) metrics. 

Here is the CE result of **MathBench**, **MathBench-A** demonstrates the performance of the model on application problems

**Models** | **Arith** | **Primary** | **Middle** | **High** | **College** | **Avg.** 
---|---|---|---|---|---|---
*Closed-source Models* | | | | | |
GPT-3.5-Turbo-0125 | 72.7 | 72.3 | 27.3 | 18.3 | 14.3 | 41.0 
GLM4 | 61.7 | 80.0 | 55.7 | 38.7 | 20.7 | 51.3 
GPT-4-0125-Preview | 76.0 | 82.3 | 59.0 | 41.3 | 35.3 | 58.8 
Qwen-Max-0428 | 72.3 | 86.3 | 65.0 | 45.0 | 27.3 | 59.2 
DeepSeek-V2-API | 82.7 | **89.3** | 59.0 | 39.3 | 29.3 | 59.9 
Claude-3-Opus | **85.7** | 85.0 | 58.0 | 42.7 | 43.7 | 63.0 
GPT-4o-2024-05-13 | 77.7 | 87.7 | **76.3** | **59.0** | **54.0** | **70.9** 
*Open-source Chat Models* | | | | | |
Yi-6B-Chat | 35.3 | 36.3 | 7.0 | 3.0 | 4.3 | 17.2 
ChatGLM3-6B | 38.0 | 41.0 | 13.7 | 5.3 | 1.7 | 19.9 
DeepSeek-7B-Chat | 48.3 | 47.7 | 8.7 | 4.3 | 2.7 | 22.3 
Qwen-7B-Chat | 50.7 | 50.7 | 22.0 | 9.3 | 6.0 | 27.7 
InternLM2-Chat-7B | 52.0 | 66.3 | <u>30.0</u> | <u>13.7</u> | <u>8.7</u> | <u>34.1</u> 
Llama-3-8B-Instruct | <u>54.7</u> | <u>71.0</u> | 25.0 | <u>19.0</u> | <u>14.0</u> | <u>36.7</u> 
Baichuan2-13B-Chat | 40.0 | 44.7 | 13.7 | 4.7 | 1.7 | 20.9 
Yi-34B-Chat | 50.7 | 62.0 | 23.0 | 14.7 | 7.7 | 31.6 
Qwen-14B-Chat | <u>63.7</u> | 61.7 | <u>39.0</u> | 21.0 | 12.0 | 39.5 
InternLM2-Chat-20B | 62.3 | <u>72.7</u> | 37.7 | <u>24.7</u> | <u>13.0</u> | <u>42.1</u> 
DeepSeek-67B-Chat | 62.0 | 72.7 | 33.3 | 21.3 | 12.0 | 40.3 
Qwen-72B-Chat | **72.0** | 71.7 | 53.7 | 32.0 | 19.0 | 49.7 
Llama-3-70B-Instruct | 70.3 | **86.0** | 53.0 | 38.7 | **34.0** | 56.4 
Qwen1.5-110B-Chat | 70.3 | 82.3 | **64.0** | **47.3** | 28.0 | **58.4** 
*Mathematical Models* | | | | | |
MammoTH-7B | 27.0 | 24.3 | 2.7 | 1.7 | 0.7 | 11.3 
MammoTH-13B | 35.0 | 43.0 | 5.0 | 4.7 | 5.0 | 18.5 
MammoTH-70B | 35.7 | 60.0 | 11.0 | 10.7 | 6.0 | 24.7 
Metamath-Llemma-7B | 51.7 | 51.0 | 8.3 | 8.3 | 5.0 | 24.9 
InternLM2-Chat-Math-7B | 53.7 | 67.0 | 41.3 | 18.3 | 8.0 | 37.7 
DeepSeek-Math-7B-Instruct | 61.0 | 74.0 | 30.3 | 24.7 | 14.3 | 40.9 
InternLM2-Chat-Math-20B | 58.7 | 70.0 | 43.7 | 24.7 | 12.7 | 41.9 
DeepSeek-Math-7B-RL | <u>68.0</u> | <u>83.3</u> | <u>44.3</u> | <u>33.0</u> | <u>23.0</u> | <u>50.3</u> 



**MathBench-T** demonstrates the performance of the model on theoretical problems

**Models** | **Primary** | **Middle** | **High** | **College** | **Avg.** 
---|---|---|---|---|---
*Closed-source Models* | | | | |
GPT-3.5-Turbo-0125 | 70.1 | 56.7 | 47.3 | 52.5 | 56.7 
GLM4 | 88.6 | 79.5 | 63.7 | 60.6 | 73.1 
GPT-4-0125-Preview | 87.2 | 81.0 | 72.0 | 73.3 | 78.4 
Claude-3-Opus | 86.0 | 79.0 | 72.6 | 77.4 | 78.7 
DeepSeek-V2-API | 88.9 | 83.7 | 70.3 | 76.3 | 79.8 
Qwen-Max-0428 | 90.4 | 83.2 | 73.4 | 74.8 | 80.4 
GPT-4o-2024-05-13 | **92.2** | **88.3** | **82.0** | **85.6** | **87.0** 
*Open-source Chat Models* | | | | |
DeepSeek-7B-Chat | 33.3 | 26.0 | 14.4 | 13.6 | 21.8 
ChatGLM3-6B | 41.6 | 32.4 | 20.2 | 12.0 | 26.6 
Yi-6B-Chat | 48.0 | 33.5 | 21.8 | 23.9 | 31.8 
Qwen-7B-Chat | 53.1 | 43.5 | 32.9 | 31.2 | 40.2 
Llama-3-8B-Instruct | 60.2 | 51.3 | 43.5 | 53.6 | 52.1 
InternLM2-Chat-7B | <u>67.3</u> | <u>55.8</u> | <u>45.4</u> | <u>42.7</u> | <u>52.8</u> 
Baichuan2-13B-Chat | 45.4 | 36.9 | 24.1 | 21.0 | 31.9 
InternLM2-Chat-20B | 64.5 | 56.2 | <u>49.9</u> | 43.2 | 53.4 
Yi-34B-Chat | 70.9 | 57.0 | 43.6 | 46.8 | 54.6 
Qwen-14B-Chat | <u>71.6</u> | <u>64.0</u> | 49.7 | 49.4 | <u>58.7</u> 
DeepSeek-67B-Chat | 78.1 | 65.7 | 55.6 | 64.6 | 66.0 
Llama-3-70B-Instruct | 71.4 | 64.3 | 62.1 | 71.2 | 67.2 
Qwen-72B-Chat | 90.9 | 80.9 | 67.1 | 69.8 | 77.2 
Qwen-1.5-110B-Chat | **93.4** | **85.0** | **76.5** | **81.5** | **84.1** 
*Mathematical Models* | | | | |
MammoTH-7B | 11.6 | 9.1 | 8.4 | 6.3 | 8.8 
MammoTH-13B | 27.5 | 18.6 | 15.0 | 17.1 | 19.5 
MetaMath-Llemma-7B | 36.6 | 33.5 | 28.8 | 25.9 | 31.2 
MammoTH-70B | 58.1 | 47.1 | 39.3 | 44.6 | 47.3 
InternLM2-Chat-Math-7B | 65.6 | 60.2 | 51.7 | 46.5 | 56.0 
DeepSeek-Math-7B-Instruct | 73.3 | 58.4 | 49.3 | 50.3 | 57.8 
InternLM2-Chat-Math-20B | 73.2 | 70.5 | 60.6 | 53.0 | 64.3 
DeepSeek-Math-7B-RL | <u>79.6</u> | <u>72.0</u> | <u>61.3</u> | <u>68.7</u> | <u>70.4</u> 


## 🔊Average Application Scores with Stages
 Models exhibit similar performances in Arithmetic and Primary stages, while demonstrating a clear performance decline from Primary to College stages.
<div align="center">
 <img src="https://github.com/open-compass/MathBench/assets/28834990/f7d83014-f4c1-45d5-bf3b-386c95c032f9" width="800"/>
</div>

## Bilingual Performance

## 📊Model Size vs. Average Score

The comparison chart of model parameter size versus performance
on MathBench for selected representative models, with
models from the same series connected by lines of the
same color. The horizontal red dotted line represents
the score of GPT-4-0125-Preview.
<div align="center">
 <img src="https://github.com/open-compass/opencompass/assets/28834990/f00ec39b-5c8f-4990-82fc-7fca826c3c64" width="800"/>
</div>

## 📈Bilingual Performance
Below is a comparison of the bilingual results of numerous Chat models on MathBench, sorted in ascending order of language average scores. 
![image](https://github.com/open-compass/opencompass/assets/28834990/38ff010f-a2c8-440d-a1f9-671f9b438957)




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
2. Prepare the dataset, you can download the data from [release file](https://github.com/open-compass/MathBench/releases/tag/v0.1.0)
```
# Download dataset from release file and copy to data/ folder
mkdir data
cp -rf mathbench_v1 ./data/ 
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
--summarizer summarizers.groups.mathbench_v1_2024 # Summarizer for MathBench-T and MathBench-A
```
If your want to see bilingual results for MathBench-A&T, replace `summarizers.groups.mathbench_v1_2024` with `summarizers.groups.mathbench_v1_2024_lang`.


# Citation and Tech Report
If you use MathBench in your research, please cite the following paper:
```
@misc{liu2024mathbench,
      title={MathBench: Evaluating the Theory and Application Proficiency of LLMs with a Hierarchical Mathematics Benchmark}, 
      author={Hongwei Liu and Zilong Zheng and Yuxuan Qiao and Haodong Duan and Zhiwei Fei and Fengzhe Zhou and Wenwei Zhang and Songyang Zhang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2405.12209},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
