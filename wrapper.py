import sys
import os
import torch
import torchaudio
import torch.nn.functional as F
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model


class GAMAWrapper:
    """
    GAMAWrapper is a class that wraps the GAMA model for inference.
    It handles loading the model, preprocessing audio data, and postprocessing output.
    """
    def __init__(self, model_path: str):
        """
        Initialize the GAMAWrapper with the path to the model.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.prompter = None
        self.load_model()

    def load_model(self):
        """
        Load the model, tokenizer, and other resources.
        """
        base_model = "./Llama-2-7b-chat-hf-qformer"
        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        # Load the base model
        self.model = LlamaForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.float32
        )
        # Setup LoRA configuration and wrap the model
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        # Load the fine-tuned state dict
        state_dict = torch.load(self.model_path, map_location='cpu')
        _ = self.model.load_state_dict(state_dict, strict=False)
        
        # Set model configuration tokens
        self.model.config.pad_token_id = 0  # typically unknown
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        # Set model in evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

        # Load your prompter (make sure utils.prompter exists)
        from utils.prompter import Prompter
        self.prompter = Prompter('alpaca_short')

    def load_audio(self, filename):
        """
        Load an audio file, preprocess it, and extract features.
        """
        waveform, sr = torchaudio.load(filename)
        audio_info = (
            f"Original input audio length {waveform.shape[1]/sr:.2f} seconds, "
            f"number of channels: {waveform.shape[0]}, sampling rate: {sr}."
        )
        # Resample if necessary
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            sr = 16000
        waveform = waveform - waveform.mean()
        # Compute filter bank features
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning',
            num_mel_bins=128, dither=0.0, frame_shift=10
        )
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            # Use functional padding for 2D tensor: pad bottom (i.e. add extra frames)
            fbank = F.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:target_length, :]
        # Normalize the features
        fbank = (fbank + 5.081) / 4.4849
        return fbank, audio_info

    def predict(self, audio_path, question):
        """
        Generate an output given an optional audio file and a language question.
        """
        instruction = question
        prompt = self.prompter.generate_prompt(instruction, None)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Check if a valid audio path is provided
        if audio_path and audio_path.lower() != 'empty':
            cur_audio_input, audio_info = self.load_audio(audio_path)
            cur_audio_input = cur_audio_input.unsqueeze(0).to(self.device)
        else:
            cur_audio_input = None
            audio_info = 'Audio is not provided; answering a pure language question.'

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            max_new_tokens=400,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            num_return_sequences=1
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                audio_input=cur_audio_input,  # Ensure your model supports this parameter!
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )
        s = generation_output.sequences[0]
        decoded = self.tokenizer.decode(s)
        # Remove the prompt from the output (adjust slicing as needed)
        output = decoded[len(prompt):].strip()
        return audio_info, output

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    model_path = 'path_to_eval/pytorch_model.bin'  # Change to your model path
    wrapper = GAMAWrapper(model_path)
    
    # Provide an actual file path or use 'empty' to indicate no audio
    audio_path = ""  # For no audio, you might set this to '' or 'empty'
    question = "Describe the audio."
    
    audio_info, answer = wrapper.predict(audio_path, question)
    print(audio_info)
    print(answer)