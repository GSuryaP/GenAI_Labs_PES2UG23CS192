# Unit 1 Assignment: The Model Benchmark Challenge

**Objective**: In this assignment, you will step beyond simply using a model and instead evaluate the architectural differences between **BERT**, **RoBERTa**, and **BART**. You will force these models to perform tasks they might not be designed for, to observe why architecture matters.

**Instructions**:
1.  Create a new Jupyter Notebook (e.g., `Unit1_Benchmark.ipynb`).
2.  Install/Import `transformers` and `pipeline`.
3.  Complete the 3 Experiments listed below using **all three models** for each task.
4.  Fill out the **Observation Table** with your results.

---

### The Models to Test
1.  **BERT** (`bert-base-uncased`): An **Encoder-only** model (designed for understanding, not generation).
2.  **RoBERTa** (`roberta-base`): An optimized **Encoder-only** model.
3.  **BART** (`facebook/bart-base`): An **Encoder-Decoder** model (designed for seq2seq tasks like translation/generation).

---

### Experiment 1: Text Generation
**Task**: Try to generate text using the prompt: `"The future of Artificial Intelligence is"`
*   **Code Hint**: `pipeline('text-generation', model='...')`
*   **Hypothesis**: Which models will fail? Why? (Hint: Can an Encoder *generate* new tokens easily?)

### Experiment 2: Masked Language Modeling (Missing Word)
**Task**: Predict the missing word in: `"The goal of Generative AI is to [MASK] new content."`
*   **Code Hint**: `pipeline('fill-mask', model='...')`
*   **Hypothesis**: This is how BERT/RoBERTa were trained. They should perform well.

### Experiment 3: Question Answering
**Task**: Answer the question `"What are the risks?"` based on the context: `"Generative AI poses significant risks such as hallucinations, bias, and deepfakes."`
*   **Code Hint**: `pipeline('question-answering', model='...')`
*   **Note**: Using a "base" model (not fine-tuned for SQuAD) might yield random or poor results. Observe this behavior.

---

### Deliverable: Observation Table

Copy this markdown table into your notebook and fill it out based on your experiments.

| Task | Model | Classification (Success/Failure) | Observation (What actually happened?) | Why did this happen? (Architectural Reason) |
| :--- | :--- | :--- | :--- | :--- |
| **Generation** | BERT | *Failure* | *Example: Generated nonsense or random symbols.* | *BERT is an Encoder; it isn't trained to predict the next word.* |
| | RoBERTa | *Failure* | Failed or produced incoherent output similar to BERT. | RoBERTa is also encoder-only and trained only for masked token prediction, not sequence generation. |
| | BART | *Success* | Generated a coherent continuation about the future of AI. | BART is an encoder–decoder model trained for seq2seq generation tasks like summarization and translation. |
| **Fill-Mask** | BERT | *Success* | *Predicted 'create', 'generate'.* | *BERT is trained on Masked Language Modeling (MLM).* |
| | RoBERTa | *Success* | Predicted similar high-quality words like "create" and "generate". | RoBERTa is an optimized MLM model with better pretraining data and training strategy. |
| | BART | *Partial Success* | Worked only when using `<mask>` token instead of `[MASK]`. | BART uses a denoising objective and different special tokens. |
| **QA** | BERT | *Partial Success* | Sometimes returned a relevant phrase like "hallucinations, bias, and deepfakes" but with low confidence. | BERT-base is not fine-tuned for QA; it lacks task-specific supervision. |
| | RoBERTa | *Partial Success* | Gave somewhat relevant answer spans but unstable or low confidence. | Also not fine-tuned for QA. Encoder-only but needs QA fine-tuning (e.g., on SQuAD). |
| | BART | *Partial Failure* | Returned strange or incomplete answers or failed. | BART-base is generative and not trained for extractive QA tasks. |

---
