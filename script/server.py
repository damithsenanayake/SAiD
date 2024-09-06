from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel


import os
from diffusers import DDIMScheduler
import torch
from said.model.diffusion import SAID_UNet1D
from said.util.audio import fit_audio_unet, load_audio
from said.util.blendshape import (
    load_blendshape_coeffs,
    save_blendshape_coeffs,
    save_blendshape_coeffs_image,
)
from dataset.dataset_voca import BlendVOCADataset


global said_model 
global model_ready

model_ready = False

def init_said_model(device="cuda:0", prediction_type="epsilon", unet_feature_dim=-1, weights_path="/home/damith/Documents/SAiD/model/SAiD/SAiD.pth"):
    global said_model
    global model_ready
    
    said_model = SAID_UNet1D(
        noise_scheduler=DDIMScheduler,
        feature_dim=unet_feature_dim,
        prediction_type=prediction_type,
    )
    said_model.load_state_dict(torch.load(weights_path, map_location=device))
    said_model.to(device)
    said_model.eval()
    
    model_ready = True
    print("SAiD Model Initiated Successfully")

def animate(
    audio_path="../BlendVOCA/audio/FaceTalk_170731_00024_TA/sentence01.wav",
    num_steps=5,
    strength=1.0,
    guidance_scale=2.0,
    guidance_rescale=0.0,
    eta=0.0,
    fps=60,
    divisor_unet=1,
    device="cuda:0"
):
    global model_ready

    if not model_ready:
        init_said_model()

    global said_model

    show_process = True

    # Load data
    waveform = load_audio(audio_path, said_model.sampling_rate)

    # Fit the size of waveform
    fit_output = fit_audio_unet(waveform, said_model.sampling_rate, fps, divisor_unet)
    waveform = fit_output.waveform
    window_len = fit_output.window_size

    # Process the waveform
    waveform_processed = said_model.process_audio(waveform).to(device)

    # Inference
    with torch.no_grad():
        output = said_model.inference(
            waveform_processed=waveform_processed,
            init_samples=None,
            mask=None,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            eta=eta,
            save_intermediate=False,
            show_process=show_process,
        )

    result = output.result[0, :window_len].cpu().numpy().tolist()

    a2f_object = {"facsNames": BlendVOCADataset.default_blendshape_classes, "weightMat": result}

    return a2f_object

app = FastAPI()

class AudioInput(BaseModel):
    audio_path: str

@app.post("/generate_blendshapes")
async def generate_blendshapes(audio_input: AudioInput):
    audio_path = audio_input.audio_path
    print("Processing audio: "+ audio_path)
    # Call the main function with the provided audio path
    try:
        a2f_object = animate(
            audio_path=audio_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return a2f_object

# To run the server on port 8011
if __name__ == "__main__":
    import uvicorn
    init_said_model()
    uvicorn.run(app, host="0.0.0.0", port=8011)
