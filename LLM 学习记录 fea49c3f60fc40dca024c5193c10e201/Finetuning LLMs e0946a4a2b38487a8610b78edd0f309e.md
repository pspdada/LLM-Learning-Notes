# Finetuning LLMs

课程：[Finetuning LLMs](https://www.bilibili.com/video/BV1Dm4y157Dc/)

## P2 Why FineTune

- What is finetuning
    
    What does mnetuning do for the model?
    
    - Lets you put more data into the model than what fits into the prompt
    - Gets the model to learn the data, rather than just get access to it
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled.png)
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%201.png)
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%202.png)
    
    - Steers the model to more consistent outputs
    - Reduces hallucinations (幻觉)
    - Customizes the model to a specific use case
    - Process is similar to the model's earlier training
- Prompt Engineering vs. Finetuning
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%203.png)
    
- Benefits of finetuning your own LLM
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%204.png)
    
- What we'll be using to finetune
    - Low level: Pytorch (Meta)
    - Higher level: Huggingface
    - Much higher level: Llama library (Lamini)

## P3 Where finetuning fits in

- Pretraining
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%205.png)
    
- Finetuning after pretraining
    - Finetuning usually refers to training further
        - Can also be self-supervised unlabeled data
        - Can be "labeled" data you curated
        - Much less data needed
        - Tool in your toolbox
    - Finetuning for generative tasks is not well-defined:
        - Updates entire model, not just part of it
        - Same training objective: next token prediction
        - More advanced ways reduce how much to update (more later!)
- What is finetuning doing for you?
    
    Both behavior change and gain knowledge
    
    - Behavior change
        - Learning to respond more consistently
        - Learning to focus, e.g. moderation
        - Teasing out capability, e.g. better at conversation
    - Gain knowledge
        - Increasing knowledge of new specific concepts
        - Correcting old incorrect information
- Tasks to finetune
    - Just text-in, text-out:
        - Extraction: text in, less text out
            - “Reading”
            - Keywords, topics, routing, agents (planning. reasoning, self-critic, tool use), etc.
        - Expansion: text in, more text out
            - "writing”
            - Chat, write emails, write code
    - Task clarity is key indicator of success
        - Clarity means knowing what's bad output vs. good output
- First time finetuning
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%206.png)
    

## P4 Instruction Finetuning

- What is instruction finetuning?
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%207.png)
    

## P5 Data preparation

- What kind of data
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%208.png)
    
- Steps to prepare your data
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%209.png)
    
- Tokenizing your data
is taking your text data and turning that intu numbers
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%2010.png)
    
    ```python
    def tokenize_function(examples):
        if "question" in examples and "answer" in examples:
            text = examples["question"][0] + examples["answer"][0]
        elif "input" in examples and "output" in examples:
            text = examples["input"][0] + examples["output"][0]
        else:
            text = examples["text"][0]
    
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
        )
    
        max_length = min(tokenized_inputs["input_ids"].shape[1], 2048)
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length
        )
        
        return tokenized_inputs
    ```
    
    这是一段Python代码，用于对输入的文本数据进行分词和处理。具体来说，它首先检查输入的数据中是否包含"question"和"answer"或"input"和"output"字段，然后将这些字段中的文本合并在一起。如果没有找到这些字段，则直接使用"data"[0]作为输入文本。
    
    接下来，该代码设置tokenizer的pad_token为eos_token，并调用tokenizer函数对文本进行分词并返回一个numpy数组。同时，设置了padding=True以确保所有样本长度相同。
    
    然后，计算得到的tokenized_inputs["input_ids"]的最大长度（不超过2048），并将truncation_side设为"left"，以便在需要截断时从左向右截取文本。
    
    最后再次调用tokenizer函数，传入max_length参数来限制输出序列的最大长度，并返回处理后的tokenized inputs。
    

## P6 Training process

## P7 Evaluation and iteration

- Evaluating generative models is notoriously difficult
    - Human expert evaluation is most reliable
    - Good test data is crucial
        - High-quality
        - Accurate
        - Generalized
        - Not seen in training data
    - Elo comparisons also popular
- LLM Benchmarks: Suite of Evaluation Methods
    
    Common LLM benchmarks:
    
    - ARC is a set of grade-school questions.
    - HellaSwag is a test of common sense.
    - MMLU is a multitask metric covering elementary math, US history, computer science, law, and more.
    - TruthfulQA measures a model's propensity to reproduce falsehoods commonly found online.
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%2011.png)
    
- Error Analysis
    - Understand base model behavior before finetuning
    - Categorize errors: iterate on data to fix these problems  in data space
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%2012.png)
    

## P8 Considerations on getting started now

- Practical approach to finetuning
    - Figure out your task
    - Collect data related to the task's inputs/outputs
    - Generate data if you don't have enough data
    - Finetune a small model (e.g.400M-1B)
    - Vary the amount of data you give the model
    - Evaluate your LLM to know what's going well vs. not
    - Collect more data to improve
    - increase task complexity
    - increase model size for performance
- Model Sizes x Compute
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%2013.png)
    
- PEFT: Parameter-Efficient Finetuning
    - LORA: Low-Rank Adaptation of LLMs
        - Fewer trainable parameters: for GPT3, 10000x less
        - Less GPU memory: for GPT3, 3x less
        - Slightly below accuracy to finetuning
        - Same inference latency
    - Train new weights in some layers, freeze main weights
        - New weights: rank decomposition matrices of original weights' change
        - At inference, merge with main weights
    - Use LoRA for adapting to new, different tasks
    
    ![Untitled](Finetuning%20LLMs%20e0946a4a2b38487a8610b78edd0f309e/Untitled%2014.png)