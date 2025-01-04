import tqdm
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import os
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from opencoconut import AutoCoconutForCausalLM, CoTDataset, split_sequences


def extract_generated_answer(model_output: str, eos_token="<|im_end|>"):
    answer_prefix = "Answer: "
    start_index = model_output.find(answer_prefix)

    if start_index == -1:
        return None

    start_index += len(answer_prefix)
    end_index = model_output.find(eos_token, start_index)

    if end_index == -1:
        return None

    extracted_answer = model_output[start_index:end_index].strip()
    return extracted_answer


@torch.no_grad()
def evaluate(
    dataloader,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_new_tokens: int,
    rank: int,
):
    total_instances = 0
    total_correct = 0
    for batch in tqdm.tqdm(dataloader):
        (
            thought_ids,
            language_ids,
            thought_mask,
            _,
            _,
            _,
        ) = split_sequences(**batch, coconut_config=model.coconut_config)
        batch_size = thought_ids.shape[0]
        total_instances += batch_size

        # Generate
        beam_output = model.generate(
            input_ids=thought_ids.to(rank),
            attention_mask=thought_mask.to(rank),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Evaluate
        for thought_ids_batch, output_batch in zip(thought_ids, beam_output):
            decoded_language_ids = tokenizer.decode(language_ids[0])
            decoded_pred_text = tokenizer.decode(output_batch)
            answer = extract_generated_answer(
                decoded_language_ids, eos_token=tokenizer.eos_token
            )
            pred_answer = extract_generated_answer(
                decoded_pred_text, eos_token=tokenizer.eos_token
            )
            if answer == pred_answer:
                total_correct += 1
            print(
                f"Input: {tokenizer.decode(thought_ids_batch, skip_special_tokens=True)}\n"
                f"Target: {answer}\n"
                f"Predicted: {pred_answer}\n"
            )
    accuracy = (total_correct / total_instances) * 100
    return accuracy


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def run(rank, world_size):
    #can't use torch.nn.parallel.DataParallel because of generate() != __call__()
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_file", type=str)
    args = parser.parse_args()
     
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model
    model = AutoCoconutForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(rank)
    model.tokenizer = tokenizer
    
    with torch.inference_mode():

        # Load data
        dataset = CoTDataset(
            "./dataset_spp/gsm8k_synthetic_cot",
            tokenizer,
            max_length=args.max_new_tokens,
            coconut_config=model.coconut_config,
            current_stage=model.coconut_config.stages,
            split="valid",
        )
        chunk_size = len(dataset)//world_size 
        chunk_size = chunk_size + 1 if chunk_size*world_size < len(dataset) else chunk_size 
        sampler_indices = [list(range(chunk_size*idx, min(len(dataset), chunk_size*(idx+1)))) for idx in range(world_size)][rank]
        print(f"Rank {rank}: ", sampler_indices) 
        subset_dataset = torch.utils.data.Subset(dataset, sampler_indices)
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        accuracy = evaluate(dataloader, tokenizer, model, args.max_new_tokens, rank=rank)
        
        # print(f"Accuracy at rank {rank}: {accuracy} %")
       # acc = torch.tensor([0], device=torch.device(f"cuda:{rank}"))
        #if rank == 0:
        
        # dist.all_reduce(acc, op=dist.ReduceOp.AVG)
       
       
        open(f"./log_file_{args.model_name.replace('/', '_')}", "a").write(f"Rank: {rank}, Acc: {accuracy}\n")       
        
        # if rank == 0: 
        #     print(torch.cuda.memory_summary(device=f"cuda:{rank}"))
        #     print("Accuracy: ", acc)
        # dist.destroy_process_group()
def main():
    world_size = 6
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
def main_debug():
    world_size = 6
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
if __name__ == "__main__":
    main_debug()