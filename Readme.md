# Final Project of NYCU.EdgeAI - Group 22

### The Implementation of Distilling Large Language Model

- **Goal**: We use the distillation method to train a student model that produces outputs similar to a teacher model. This allows for higher inference throughput by leveraging the smaller size of the student model.

- **Methodology**: We follow a teacher-student training pipeline. First, we generate the teacher model's (**Llama3.2-3B-Instruct**) output and train the student model (**TinyLlama-1.1B**) to mimic it. Note that the training strategy follows **LoRA**.

- **Result**: Achieved **82.720 tokens/second** throughput with a **7.694 perplexity (PPL)**.

---

## Getting Started

We separate our code into two stages: training the student model (`teacher_student.py`) and evaluating the model throughput (`result.py`).

### Requirements

- Python 3.x  
- Install dependencies using:
  ```bash
  pip install -r environment.txt

### Training student model under LoRA
    
    python teacher_student.py
    
After this stage, a directory named distilled_student_lora/ will be created, which contains the distilled student model weights.


### Evaluating
    python result.py

The results of our submission were obtained on a Google Colab T4 environment (due to difficulty connecting to TA's server).

---

## References

https://pytorch.org/blog/llama-into-torchtune/

https://github.com/KylinC/Llama-3-Distill
