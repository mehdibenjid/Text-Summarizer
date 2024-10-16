import evaluate
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
    
    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches for processing."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]
    
    def calculate_metric_on_test_ds(self, dataset, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="article", column_summary="highlights"):
        
        # Initialize lists to store predictions and references
        predictions = []
        references = []
        
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt").to(device)
            
            summaries = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], 
                length_penalty=0.8, num_beams=8, max_length=128
            )
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]
            
            predictions.extend(decoded_summaries)
            references.extend(target_batch)
        
        # Calculate ROUGE scores using the `evaluate` library
        rouge = evaluate.load('rouge')
        score = rouge.compute(predictions=predictions, references=references,
                              rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_aggregator=True)
        
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        
        # Compute ROUGE scores on a subset
        score = self.calculate_metric_on_test_ds(
            dataset['test'][0:10], model, tokenizer, batch_size=2, 
            column_text='dialogue', column_summary='summary'
        )

        # Create a DataFrame and save the results
        rouge_dict = {key: [score[key]] for key in score.keys()}
        df = pd.DataFrame(rouge_dict)
        df.to_csv(self.config.metric_file_name, index=False)
