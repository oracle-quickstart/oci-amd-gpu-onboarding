# ML Fine-Tuning With PyTorch & ROCM on AMD Instinct MI300X

## Overview

- Procedure for  fine-tuning using the LoRA (Low-Rank Adaptation) approach with the Hugging Face library. We will utilize FSDP (Fully Sharded Data Parallel) for efficient distributed training. 
- A customized Docker container was utilized, provided by AMD for this specific purpose. However, for those seeking to replicate the environment, a comprehensive list of necessary packages along with detailed setup instructions is available. This will enable you to create a suitable environment for conducting the finetuning exercise independently. 

## Setting up the Virtual Enviroment 
### 1. Clone the Repository (Optional)
- If you have not cloned this repo already, please proceed to do it 

```bash
   git clone https://github.com/oracle-quickstart/oci-amd-gpu-onboarding.git
   cd oci-amd-gpu-onboarding
```
### 2. Create virtual environment names venv_ft
```bash
python3 -m venv venv
```
### 3. Activate the virtual enviroment 
```bash
source vnenv/bin/activate
```

### 4. Install required packages
```bash
pip install -r requirements.txt
```
- You are all set to perform fine-tuning!

## Execute Fine-Tuning

To start the fine-tuning process, run the following command:

```bash
ACCELERATE_USE_FSDP=1 torchrun --nproc_per_node=8 ./lora.py --config llama_3_70b_fsdp_lora.yaml
```

### Parameters:
- `ACCELERATE_USE_FSDP=1`: Enables Fully Sharded Data Parallelism.
- `torchrun --nproc_per_node=8`: Runs the training across 8 processes (GPUs).
- `./lora.py`: The script used for training.
- `--config llama_3_70b_fsdp_lora.yaml`: Configuration file for the LoRA fine-tuning.

## Configuration File

Before running the training script, you need to modify the `llama_3_70b_fsdp_lora.yaml` configuration file. Here are the key parameters to update:

1. **model_id**: Set this to the directory where your Hugging Face model checkpoints are stored.
   ```yaml
   model_id: "path/to/your/huggingface/model/checkpoints"
   ```

2. **max_seq_length**: Update this to the desired sequence length for training.
   ```yaml
   max_seq_length: <your_sequence_length>
   ```

3. **per_device_train_batch_size**: Adjust the batch size based on the `max_seq_length`:
   - For `max_seq_length=1024`: set `per_device_train_batch_size: 9`
   - For `max_seq_length=2048`: set `per_device_train_batch_size: 4`
   - For `max_seq_length=4096`: set `per_device_train_batch_size: 1`

   Example:
   ```yaml
   per_device_train_batch_size: 9  # for max_seq_length=1024
   ```

## Example Configuration

Here’s an example of how your `llama_3_70b_fsdp_lora.yaml` might look after editing:

```yaml
model_id: "path/to/your/huggingface/model/checkpoints"
max_seq_length: 1024
per_device_train_batch_size: 9
```

## Running the Fine-Tuning

After making the necessary changes to the configuration file, you can execute the fine-tuning command provided above.

## Monitoring and Logging

Monitor the training logs for any errors or warnings. Adjust the configuration as needed based on the performance and resource utilization. We used mlflow to track experiement. Here is the link to our [experiement](http://mlflow-benchmarking.corrino-oci.com:5000/#/experiments/63/runs/c836a0297f41440aab97fa55c714d7e4)

