import os
import pandas as pd
import torch
import evaluate
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Custom Dataset Class to load images from a local Kaggle folder
class KaggleHandwritingDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # The CSV must have 'file_name' and 'text' columns
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        # Load image
        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        
        # Process image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length,
            truncation=True
        ).input_ids
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def compute_metrics(pred):
    cer_metric = evaluate.load("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def main():
    print("=== TrOCR Fine-Tuning Pipeline (Kaggle Edition) ===")
    
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Detected Hardware Accelerator: [{device.upper()}]")

    # 2. Dataset Setup
    # INSTRUCTIONS: Download your dataset from Kaggle and place the images in 'dataset/images/'
    # and create a 'dataset/labels.csv' with columns: file_name, text
    dataset_dir = "./dataset"
    images_dir = os.path.join(dataset_dir, "images")
    csv_file = os.path.join(dataset_dir, "labels.csv")
    
    if not os.path.exists(csv_file) or not os.path.exists(images_dir):
        print(f"\n[Error] Dataset not found! Please create '{images_dir}' and '{csv_file}'.")
        print("The labels.csv should have two columns: 'file_name' and 'text'.")
        print("Example row: image_001.jpg, Embrace the good moments")
        return

    print("Loading local dataset...")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} images in labels.csv")

    # 3. Load Base Model and Processor
    model_name = "microsoft/trocr-base-handwritten"
    print(f"Loading base model: {model_name}")
    global processor 
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Set special tokens for Seq2Seq processing
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # 4. Prepare PyTorch Dataset
    train_dataset = KaggleHandwritingDataset(root_dir=images_dir, df=df, processor=processor)

    # 5. Training Arguments (Ready for overnight training)
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=4, # Adjust based on your Mac's RAM
        fp16=False, 
        output_dir="./fine_tuned_trocr",
        logging_steps=10,
        save_steps=100,
        num_train_epochs=5, # Train for 5 full cycles over the dataset
        use_mps_device=(device == "mps"),
        report_to="none"
    )

    # 6. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # 7. Train and Save
    print("\nStarting Fine-Tuning Loop... (This may take hours depending on dataset size)")
    trainer.train()
    
    print("\nTraining complete! Saving custom model to ./fine_tuned_trocr")
    trainer.save_model("./fine_tuned_trocr")
    processor.save_pretrained("./fine_tuned_trocr")
    print("Success! Your custom model will automatically be loaded next time you run main.py")

if __name__ == "__main__":
    main()
