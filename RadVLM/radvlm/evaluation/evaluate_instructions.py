import os
import json
import argparse
import random
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import PartialState
from accelerate.utils import gather_object

import re
    
from radvlm.data.utils import custom_collate_fn
from radvlm.data.datasets import (
    CheXpert_Dataset_MM,
    VinDr_CXR_Single_Label_Dataset,
    VinDr_CXR_Dataset,
    MIMIC_Dataset_MM,
    Chest_ImaGenome_Dataset,
    MS_CXR
)

from radvlm.evaluation.models_loading_inference import load_model_and_processor, inference_radialog, inference_llavamed, inference_llavaov, inference_chexagent, inference_maira2_report, inference_maira2_grounding, inference_qwen2vl, inference_qwen2vl_vllm, inference_medgemma_4b_pt, inference_llava_rad, inference_deepmedix_r1
from radvlm.evaluation.utils import plot_images_with_Bbox
from radvlm.evaluation.compute_metrics_tasks import evaluate_results

from radvlm import DATA_DIR
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(script_dir, "results")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process inference for a single instruction")
    parser.add_argument('--task', type=str, required=True, choices=[
        "abnormality_classification",
        "abnormality_grounding",
        "abnormality_detection",
        "report_generation",
        "region_grounding",
        "object_grounding", 
        "phrase_grounding",
        "vqa"
    ], help='The task to perform')
    parser.add_argument('--model_name', type=str, required=True, help='The model name to evaluate')
    parser.add_argument('--num_batches', type=int, default=None, help='Number of batches to process, if none process all')
    parser.add_argument('--r1', action='store_true', help='Enable R1 mode (default: False)')
    parser.add_argument('--vllm', action='store_true', help='Enable vllm for inference (default: False)')
    parser.add_argument('--load_outputs', type=str, default=None, help='Path to a JSON file with precomputed outputs to load')
    parser.add_argument('--temp', type=float, default=0.0, help='temperature')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='max new tokens')
    parser.add_argument('--remove_think_suffixe', action='store_true', help='remove the think suffixe')
    return parser.parse_args()
    


def load_dataset(task, data_dir):
    """
    Load the dataset based on the task.

    Args:
        task (str): The task to perform.
        data_dir (str): The base directory for data.

    Returns:
        Dataset: The loaded dataset object.
    """
    if task == "abnormality_classification":
        dataset_path = os.path.join(data_dir, "CheXpert")
        dataset = CheXpert_Dataset_MM(datasetpath=dataset_path, split="valid", flag_img=False)

    elif task == "abnormality_grounding":
        dataset_path = os.path.join(data_dir, "VinDr-CXR")
        dataset = VinDr_CXR_Single_Label_Dataset(
            datasetpath=dataset_path, split="valid", flag_img=False
        )
    elif task == "abnormality_detection":
        dataset_path = os.path.join(data_dir, "VinDr-CXR") 
        dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="valid", flag_img = False)
    elif task == "report_generation":
        datasetpath = os.path.join(data_dir, 'MIMIC-CXR-JPG')
        filtered_reports = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
        dataset = MIMIC_Dataset_MM(
            datasetpath=datasetpath,
            split="test",
            flag_img=False,
            flag_lab=True,
            only_frontal=True, # to evaluate only frontal images 
            filtered_reports_dir=filtered_reports, # or use filtered_reports 
            seed=0
        )

    elif task == "region_grounding" or task == "phrase_grounding":
        datasetpath = os.path.join(data_dir, 'MIMIC-CXR-JPG')
        datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
        split = "valid"
        if task == "region_grounding":
            dataset = Chest_ImaGenome_Dataset(
            datasetpath=datasetpath,
            datasetpath_chestima=datasetpath_chestima, 
            split=split, 
            flag_img=False, 
            flag_lab=False,
            flag_instr=True, 
            flag_txt=False, 
            seed=4
            )
        else:
            datasetpath_mimic = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
            datasetpath_mscxr = os.path.join(DATA_DIR, 'MS-CXR')

            split = "valid"
            sentencesBBoxpath = os.path.join(datasetpath_mscxr, 'sentences_and_BBox_mscxr')
            dataset = MS_CXR(
                datasetpath = datasetpath_mimic,
                split=split, flag_img=True, 
                flag_lab=True, only_frontal=True, 
                flag_instr=True, 
                sentencesBBoxpath=sentencesBBoxpath,
                seed=0)

    else:
        raise ValueError(f"Unsupported task: {task}")
    
    print(f"Dataset size: {len(dataset)}")
    return dataset


def process_inference_for_single_instruction(tokenizer, model, processor, data_loader, process_batch_num=None, model_name='llavaov', task='report_generation', r1=False, use_vllm=False, args=None):
    if use_vllm:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dataset_path = os.path.join(tmpdir, "test.parquet")
            return inference_qwen2vl_vllm(data_loader=data_loader, model_name=model_name, process_batch_num=process_batch_num, r1=r1, test_dataset_path=test_dataset_path, temperature=args.temp, max_new_tokens=args.max_new_tokens, remove_think_suffixe=args.remove_think_suffixe)


    ret = []
    total_batches = len(data_loader)
    for batch_i, batch in enumerate(data_loader):
        if process_batch_num and batch_i >= process_batch_num:
            break
        if batch_i % 10 == 0:
            print(f"Processing batch {batch_i + 1} / {total_batches}")
        datapoint = batch[0]

        prompt = datapoint["instr"]["question"]

        image_path = datapoint['img_path'] 

        if model_name =='radialog':
            if task == 'report_generation':
                prompt = "Write a radiology report for this X-Ray."
            elif task == 'abnormality_classification':
                questions_variations = [
                    "List all the findings in this report.",
                    "Enumerate the observations from the report.", 
                    "What findings can be identified from this report?"
                ]
                prompt = random.choice(questions_variations)
                
            generated_text, _ = inference_radialog(tokenizer, model, image_path, prompt)
            

        elif model_name == 'chexagent':
            if task == 'report_generation' or task == 'abnormality_classification':
                if task == 'report_generation':
                    prompt = "Write an example findings section for the CXR"
                else:
                    prompt = "Identify any diseases visible in the given CXR. Options:\n atelectasis, cardiomegaly, consolidation, edema, enlarged cardiomediastinum, fracture, lung lesion, lung opacity, pleural effusion, pleural other, pneumonia, pneumothorax, support devices"
                generated_text = inference_chexagent(model, tokenizer, image_path, prompt)

            elif task in ['abnormality_grounding', 'phrase_grounding', 'region_grounding']:

                if task == 'abnormality_grounding':
                    questions_variations = [
                        "Detect {} in the given image.",
                        "Locate areas in the chest X-ray where {} is present, using bounding box coordinates",
                        "Localize {} in the bounding box format for the given image.",
                        "Find the locations of {} in the bounding box format for the given image.",
                        "Locate {} for the given image.",
                        "Examine the chest X-ray and mark the regions affected by {} with bounding boxes",
                        "Detect the following in the image: {}.",
                        "Examine the image for regions affected by {}, and indicate their positions with bounding boxes.",
                        "Perform detection for {}.",
                        "Abnormality Grounding (VinDr-CXR): {}.",
                    ]
                else:
                    questions_variations = [
                        "Please locate the following anotomical region: {}",
                        "Identify the position of the following region in the CXR: {}",
                    ]
                prompt = random.choice(questions_variations).format(datapoint["label"])
                generated_text = inference_chexagent(model, tokenizer, image_path, prompt, grounding=True)

        elif model_name == 'llavamed':
            generated_text, _ = inference_llavamed(model, processor, image_path, prompt)

        elif model_name == 'maira2':
            if task == 'report_generation':
                generated_text = inference_maira2_report(model, processor, image_path, prompt)
            elif task == 'abnormality_grounding' or task == 'phrase_grounding' or task == 'region_grounding':
                generated_text = inference_maira2_grounding(model, processor, image_path, datapoint['label'])

        elif model_name == "google/medgemma-4b-pt":
            if task == 'report_generation':
                generated_text, prompt = inference_medgemma_4b_pt(model, processor, image_path, prompt)

            elif task == 'abnormality_grounding' or task == 'phrase_grounding' or task == 'region_grounding':
                assert False, "medgemma only implemented for report gernation TODO"

        elif model_name == "microsoft/llava-rad":
            if task == 'report_generation':
                generated_text, prompt = inference_llava_rad(model=model, tokenizer=tokenizer, image_processor=processor, image_path=image_path, prompt=prompt)

            elif task == 'abnormality_grounding' or task == 'phrase_grounding' or task == 'region_grounding':
                assert False, "medgemma only implemented for report gernation TODO"
        elif model_name == "Qika/DeepMedix-R1":
            if task == 'report_generation':
                final_answer, full_output, prompt = inference_deepmedix_r1(model=model, tokenizer=processor, image_path=image_path, prompt=prompt)
            else:
                assert False, "Qika/DeepMedix-R1 only supports report generation"
        else:
            if r1:
                assert False, "deprecated, use vllm"
                prompt = prompt + " /think"
            if 'qwen' in model_name.lower():
                generated_text, _ = inference_qwen2vl(model, processor, image_path, prompt)
            else:
                generated_text, _ = inference_llavaov(model, processor, image_path, prompt)

        # Store results in dictionary 
        optional_keys = ["id", "idx", "img_path", "img", "labels", "label", "txt", "boxes"]
        ans = {}

        if r1:
            assert False, "deprecated, use vllm"
            match_pattern = r'<answer>(.*?)</answer>'
            m = re.search(match_pattern, generated_text, re.DOTALL)
            if m:
                generated_text = m.group(1).strip()
            else:
                # fallback if tags missing
                generated_text = generated_text.strip()

        if model_name == "Qika/DeepMedix-R1":
            generated_text = final_answer
            ans["full_output"] = full_output

        ans["output"] = generated_text
        ans["instr"] = prompt
        ans["answer"] = datapoint['instr']["answer"]
        for key in optional_keys:
            if key in datapoint:
                ans[key] = datapoint[key]
        ret.append(ans)

    return ret


def save_results(metrics, model_name, task, num_batches, output=False):
    ensure_directory_exists(RESULTS_DIR)
    # extract just the last path component
    last_part = os.path.basename(model_name)
    # if it's a checkpoint, grab the parent directory name
    if re.match(r'^checkpoint-.*', last_part):
        model_name = os.path.basename(os.path.dirname(model_name))
    else:
        model_name = last_part
    filename = f"{model_name}_{task}"
    if output:
        filename = filename + '_output'
    if num_batches is not None:
        filename += "_partial"
    filename += ".json"
    results_path = os.path.join(RESULTS_DIR, filename)
    with open(results_path, "w") as json_file:
        json.dump(metrics, json_file, indent=2)
    print(f"Results saved to {results_path}")


def display_sample_outputs(output, num_samples=10):
    num_show_output = min(num_samples, len(output))
    for i in range(num_show_output):
        print(f"Instruction {i + 1}: {output[i]['instr']}")
        print("Prediction:")
        print(output[i]["output"])
        print("Ground truth:")
        print(output[i]["answer"])
        print("----------------------------------------------------")


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def no_vllm(args):
    distributed_state = None
    dataset = load_dataset(args.task, DATA_DIR)
    # Load dataset
    if args.load_outputs:
        with open(args.load_outputs, "r") as json_file:
            print(f"loaded outputs {args.load_outputs=}")
            output = json.load(json_file)
    else:
        tokenizer, model, processor = load_model_and_processor(args.model_name)
        distributed_state = PartialState()
                
        model.to(distributed_state.device)
        model.eval()

        
        

        # Prepare DataLoader
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_state.num_processes,
            rank=distributed_state.process_index
        )
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            collate_fn=custom_collate_fn
        )

        # Run inference

        output = process_inference_for_single_instruction(
            tokenizer,
            model,
            processor,
            data_loader,
            process_batch_num=args.num_batches,
            model_name=args.model_name, 
            task=args.task, 
            r1=args.r1
        )

        # Gather results
        distributed_state.wait_for_everyone()
        output = gather_object(output)
        if args.task == "report_generation":
            save_results(output, args.model_name, args.task, args.num_batches, output=True)


    # Evaluate and save results
    if (distributed_state is None) or distributed_state.is_main_process:
        display_sample_outputs(output)
        if args.task == "region_grounding" or args.task=="abnormality_grounding" or args.task=="phrase_grounding":
            plot_images_with_Bbox(output, num_samples=16, results_dir=RESULTS_DIR)

        metrics = evaluate_results(args.task, output, dataset)
        save_results(metrics, args.model_name, args.task, args.num_batches)
        


        
    print("Inference and evaluation complete.")

def with_vllm(args):
    dataset = load_dataset(args.task, DATA_DIR)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=custom_collate_fn
    )

    output = process_inference_for_single_instruction(
        None,
        None,
        None,
        data_loader,
        process_batch_num=args.num_batches,
        model_name=args.model_name, 
        task=args.task, 
        r1=args.r1,
        use_vllm=args.vllm,
        args=args,
    )

    if args.task == "report_generation":
        save_results(output, args.model_name, args.task, args.num_batches, output=True)

    display_sample_outputs(output)
    if args.task == "region_grounding" or args.task=="abnormality_grounding" or args.task=="phrase_grounding":
        plot_images_with_Bbox(output, num_samples=16, results_dir=RESULTS_DIR)

    metrics = evaluate_results(args.task, output, dataset)
    save_results(metrics, args.model_name, args.task, args.num_batches)
        
    print("Inference and evaluation complete.")

if __name__ == "__main__":
    args = parse_arguments()

    if args.vllm:
        with_vllm(args)
    else:
        no_vllm(args)
