# SAiD: Blendshape-based Audio-Driven Speech Animation with Diffusion

This is the official code for [SAiD: Blendshape-based Audio-Driven Speech Animation with Diffusion]().

## Installation

Run the following command to install it as a pip module:

```bash
pip install .
```

If you are developing this repo or want to run the scripts, run instead:

```bash
pip install -e .[dev]
```

If there is an error related to pyrender, install additional packages as follows:

```bash
apt-get install libboost-dev libglfw3-dev libgles2-mesa-dev freeglut3-dev libosmesa6-dev libgl1-mesa-glx
```

## Directories

- `data`: It contains data used for preprocessing and training.
- `model`: It contains the weights of VAE, which is used for the evaluation.
- `blender-addon`: It contains the blender addon that can visualize the blendshape coefficients.
- `script`: It contains Python scripts for preprocessing, training, inference, and evaluation.
- `doc`: It contains the resources for the documentation, such as sample videos.

## Inference

You can download the pretrained weights of SAiD from [Hugging Face Repo](https://huggingface.co/yunik1004/SAiD).

```bash
python script/inference.py \
        --weights_path "<SAiD_weights>.pth" \
        --audio_path "<input_audio>.wav" \
        --output_path "<output_coeffs>.csv" \
        [--init_sample_path "<input_init_sample>.csv"] \  # Required for editing
        [--mask_path "<input_mask>.csv"]  # Required for editing
```

## BlendVOCA

### Construct Blendshape Facial Model

Due to the license issue of VOCASET, we cannot distribute BlendVOCA directly.
Instead, you can preprocess `data/blendshape_residuals.pickle` after constructing `BlendVOCA` directory as follows for the simple execution of the script.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   └─ templates
      ├─ ...
      └─ FaceTalk_170915_00223_TA.ply
```

- `templates`: Download the template meshes from [VOCASET](https://voca.is.tue.mpg.de/download.php).

```bash
python script/preprocess_blendvoca.py \
        --blendshapes_out_dir "<output_blendshapes_dir>"
```

If you want to generate blendshapes by yourself, do the folowing instructions.

1. Unzip `data/ARKit_reference_blendshapes.zip`.
2. Download the template meshes from [VOCASET](https://voca.is.tue.mpg.de/download.php).
3. Crop template meshes using `data/FLAME_head_idx.txt`.
You can crop more indices and then restore them after finishing the construction process.
4. Use [Deformation-Transfer-for-Triangle-Meshes](https://github.com/guyafeng/Deformation-Transfer-for-Triangle-Meshes) to construct the blendshape meshes.
   - Use `data/ARKit_landmarks.txt` and `data/FLAME_head_landmarks.txt` as marker vertices.
   - Find the correspondance map between neutral meshes, and use it to transfer the deformation of arbitrary meshes.
5. Create `blendshape_residuals.pickle`, which contains the blendshape residuals in the following Python dictionary format.
Refer to `data/blendshape_residuals.pickle`.

    ```text
    {
        'FaceTalk_170731_00024_TA': {
            'jawForward': <np.ndarray object with shape (V, 3)>,
            ...
        },
        ...
    }
    ```

### Generate Blendshape Coefficients

You can simply unzip `data/blendshape_coeffcients.zip`.

If you want to generate coefficients by yourself, we recommend constructing the `BlendVOCA` directory as follows for the simple execution of the script.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   ├─ blendshapes_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ noseSneerRight.obj
   ├─ templates_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA.obj
   └─ unposedcleaneddata
      ├─ ...
      └─ FaceTalk_170915_00223_TA
         ├─ ...
         └─ sentence40
```

- `blendshapes_head`: Place the constructed blendshape meshes (head).
- `templates_head`: Place the template meshes (head).
- `unposedcleaneddata`: Download the mesh sequences (unposed cleaned data) from [VOCASET](https://voca.is.tue.mpg.de/download.php).

And then, run the following command:

```bash
python script/optimize_blendshape_coeffs.py \
        --blendshapes_coeffs_out_dir "<output_coeffs_dir>"
```

After generating blendshape coefficients, create `coeffs_std.csv`, which contains the standard deviation of each coefficients. Refer to `data/coeffs_std.csv`.

```text
jawForward,...
<std_jawForward>,...
```

## Training / Evaluation on BlendVOCA

### Dataset Directory Setting

We recommend constructing the `BlendVOCA` directory as follows for the simple execution of scripts.

```bash
├─ audio-driven-speech-animation-with-diffusion
│  ├─ ...
│  └─ script
└─ BlendVOCA
   ├─ audio
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ sentence40.wav
   ├─ blendshape_coeffs
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ sentence40.csv
   ├─ blendshapes_head
   │  ├─ ...
   │  └─ FaceTalk_170915_00223_TA
   │     ├─ ...
   │     └─ noseSneerRight.obj
   └─ templates_head
      ├─ ...
      └─ FaceTalk_170915_00223_TA.obj
```

- `audio`: Download the audio from [VOCASET](https://voca.is.tue.mpg.de/download.php).
- `blendshape_coeffs`: Place the constructed blendshape coefficients.
- `blendshapes_head`: Place the constructed blendshape meshes (head).
- `templates_head`: Place the template meshes (head).

### Training VAE, SAiD

- Train VAE

    ```bash
    python script/train_vae.py \
            --output_dir "<output_logs_dir>" \
            [--coeffs_std_path "<coeffs_std>.txt"]
    ```

- Train SAiD

    ```bash
    python script/train.py \
            --output_dir "<output_logs_dir>"
    ```

### Evaluation

1. Generate SAiD outputs on the test speech data

    ```bash
    python script/test_inference.py \
            --weights_path "<SAiD_weights>.pth" \
            --output_dir "<output_coeffs_dir>"
    ```

2. Remove `FaceTalk_170809_00138_TA/sentence32-xx.csv` files from the output directory.
Ground-truth data does not contain the motion data of `FaceTalk_170809_00138_TA/sentence32`.

3. Evaluate SAiD outputs: FD, WInD, and Multimodality.

    ```bash
    python script/test_evaluate.py \
            --coeffs_dir "<input_coeffs_dir>" \
            [--vae_weights_path "<VAE_weights>.pth"] \
            [--blendshape_residuals_path "<blendshape_residuals>.pickle"]
    ```

4. We have to generate the videos to compute the AV offset/confidence.
To avoid the memory leak issue of the pyrender module, we use the shell script.
After updating `COEFFS_DIR` and `OUTPUT_DIR`, run the script:

    ```bash
    # Fix 1: COEFFS_DIR="<input_coeffs_dir>"
    # Fix 2: OUTPUT_DIR="<output_video_dir>"
    python script/test_render.sh
    ```

5. Use [SyncNet](https://github.com/joonson/syncnet_python) to compute the AV offset/confidence.

## Inference Results

### Comparison with baseline methods

|| GT | SAiD (Ours) | end2end_AU_speech | VOCA+QP | MeshTalk+QP | FaceFormer+QP | CodeTalker+QP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FaceTalk_170731_00024_TA/<br>sentence01.wav| ![](doc/video/GT/FaceTalk_170731_00024_TA-sentence01.mp4){width=200} | ![](doc/video/SAiD/FaceTalk_170731_00024_TA-sentence01-0-vocaset.mp4){width=200} | ![](doc/video/end2end_AU_speech/FaceTalk_170731_00024_TA-sentence01.mp4){width=200} | ![](doc/video/VOCA/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200} | ![](doc/video/MeshTalk/FaceTalk_170731_00024_TA-sentence01.mp4){width=200} | ![](doc/video/FaceFormer/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200} | ![](doc/video/CodeTalker/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200} |
|FaceTalk_170809_00138_TA/<br>sentence02.wav| ![](doc/video/GT/FaceTalk_170809_00138_TA-sentence02.mp4){width=200} | ![](doc/video/SAiD/FaceTalk_170809_00138_TA-sentence02-0-vocaset.mp4){width=200} | ![](doc/video/end2end_AU_speech/FaceTalk_170809_00138_TA-sentence02.mp4){width=200} | ![](doc/video/VOCA/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200} | ![](doc/video/MeshTalk/FaceTalk_170809_00138_TA-sentence02.mp4){width=200} | ![](doc/video/FaceFormer/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200} | ![](doc/video/CodeTalker/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200} |

### Ablation studies

|| SAiD (Base) | train w/ squared error | train w/o velocity loss | train w/o alignment bias | finetune pre-trained Wav2Vec 2.0 |
|---|:---:|:---:|:---:|:---:|:---:|
|FaceTalk_170731_00024_TA/<br>sentence01.wav|![](doc/video/SAiD/FaceTalk_170731_00024_TA-sentence01-0-vocaset.mp4){width=200}|![](doc/video/SAiD/ablation/squared_error/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200}|![](doc/video/SAiD/ablation/velocity_loss/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200}|![](doc/video/SAiD/ablation/no_alignment_bias/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200}|![](doc/video/SAiD/ablation/finetune_wav2vec/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200}|
|FaceTalk_170809_00138_TA/<br>sentence02.wav|![](doc/video/SAiD/FaceTalk_170809_00138_TA-sentence02-0-vocaset.mp4){width=200}|![](doc/video/SAiD/ablation/squared_error/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200}|![](doc/video/SAiD/ablation/velocity_loss/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200}|![](doc/video/SAiD/ablation/no_alignment_bias/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200}|![](doc/video/SAiD/ablation/finetune_wav2vec/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200}|

### Diversity on SAiD outputs

We visualize the vertex position differences in SAiD outputs over the mean output.
We use [viridis](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html) colormap with a range of [0, 0.001].

|| Output 1 | Output 2 | Output 3 | Output 4 | Output 5 |
|---|:---:|:---:|:---:|:---:|:---:|
|FaceTalk_170731_00024_TA/<br>sentence01.wav|![](doc/video/SAiD/diversity/FaceTalk_170731_00024_TA-sentence01-0.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170731_00024_TA-sentence01-1.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170731_00024_TA-sentence01-2.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170731_00024_TA-sentence01-3.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170731_00024_TA-sentence01-4.mp4){width=200}|
|FaceTalk_170809_00138_TA/<br>sentence02.wav|![](doc/video/SAiD/diversity/FaceTalk_170809_00138_TA-sentence02-0.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170809_00138_TA-sentence02-1.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170809_00138_TA-sentence02-2.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170809_00138_TA-sentence02-3.mp4){width=200}|![](doc/video/SAiD/diversity/FaceTalk_170809_00138_TA-sentence02-4.mp4){width=200}|

### Editability of SAiD

We visualize the editing results of SAiD with two different cases:

1. Motion in-betweening
2. Motion generation with blendshape-specific constraints by masking coefficients corresponding to certain blendshapes.

Hatched boxes indicate the masked areas that should be invariant during the editing.

|| In-betweening | Blendshape-specific constraints |
|---|:---:|:---:|
|FaceTalk_170731_00024_TA/<br>sentence01.wav|<img src="doc/image/editing/FaceTalk_170731_00024_TA-sentence01-0-inbetween.png" /><br><video width="200" src="https://private-user-images.githubusercontent.com/20185342/280587030-ee826381-4ae0-4a58-8cb8-052716ec8743.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTkyNzA5MzYsIm5iZiI6MTY5OTI3MDYzNiwicGF0aCI6Ii8yMDE4NTM0Mi8yODA1ODcwMzAtZWU4MjYzODEtNGFlMC00YTU4LThjYjgtMDUyNzE2ZWM4NzQzLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTA2VDExMzcxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTExMDAwODI3ZTcyY2I2NTBiYzhiOTRjNzc2YjNmYmNhNjJkNjEzYWYyNTEwMWY2Y2QwNjFlMDM2ODI2NGJhMzEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.GlRLccdFmlz5fFE1ioAeyzL_aF7CIePe9q41sJrWA9A" />|<img src="doc/image/editing/FaceTalk_170731_00024_TA-sentence01-0-blendshape.png" /><br><video width="200" src="https://private-user-images.githubusercontent.com/20185342/8be3ab98-669b-4d3b-9ba6-4a74eca3be06" />|
|FaceTalk_170809_00138_TA/<br>sentence02.wav|<img src="doc/image/editing/FaceTalk_170809_00138_TA-sentence02-0-inbetween.png" /><br><video width="200" src="https://private-user-images.githubusercontent.com/20185342/46e07f16-1128-4ba8-ba2f-a93233ad3a44" />|<img src="doc/image/editing/FaceTalk_170809_00138_TA-sentence02-0-blendshape.png" /><br><video width="200" src="https://private-user-images.githubusercontent.com/20185342/260a2a7a-d634-4059-87f2-1b73ea3ef250" />|
<!--
|FaceTalk_170731_00024_TA/<br>sentence01.wav|![](doc/image/editing/FaceTalk_170731_00024_TA-sentence01-0-inbetween.png){width=200}<br>![](doc/video/SAiD/editing/FaceTalk_170731_00024_TA-sentence01-0-inbetween.mp4){width=200}|![](doc/image/editing/FaceTalk_170731_00024_TA-sentence01-0-blendshape.png){width=200}<br>![](doc/video/SAiD/editing/FaceTalk_170731_00024_TA-sentence01-0-blendshape.mp4){width=200}|
|FaceTalk_170809_00138_TA/<br>sentence02.wav|![](doc/image/editing/FaceTalk_170809_00138_TA-sentence02-0-inbetween.png){width=200}<br>![](doc/video/SAiD/editing/FaceTalk_170809_00138_TA-sentence02-0-inbetween.mp4){width=200}|![](doc/image/editing/FaceTalk_170809_00138_TA-sentence02-0-blendshape.png){width=200}<br>![](doc/video/SAiD/editing/FaceTalk_170809_00138_TA-sentence02-0-blendshape.mp4){width=200}|
-->

### Visualization of SAiD outputs on different blendshape facial models

Since the MetaHuman does not support the `mouthClose` blendshape, we use the editing algorithm to ensure the corresponding blendshape coefficients of the outputs are all zero.

||VOCASET - FaceTalk_170725_00137_TA | [VRoid Studio - AvatarSample_A](https://hub.vroid.com/en/characters/2287322741607496883/models/1995551907338074831) | [MetaHuman - Ada](https://www.unrealengine.com/en-US/metahuman) | [Unity_ARKitFacialCapture - Sloth](https://github.com/kodai100/Unity_ARKitFacialCapture/tree/master/Assets/Models) |
|---|:---:|:---:|:---:|:--:|
| FaceTalk_170731_00024_TA/<br>sentence01.wav | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/6e24222f-5acd-453e-894d-c604f4404b3c"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/1d060d91-d3f5-442a-8c03-0a9aa82135bd"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/d855bc97-057a-465f-9565-bb1d093932d5"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/021c0a13-eb99-49b2-b8ca-46591d21c8fb"> |
| FaceTalk_170809_00138_TA/<br>sentence02.wav | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/4fb6161c-0eb1-48b4-87c0-b644902a76bc"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/0609209a-2815-4e3a-9f69-1945d0ff0536"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/62fcccb6-4f57-45ca-9361-e68ad6835de3"> | <video width="200" src="https://private-user-images.githubusercontent.com/20185342/84a01817-18de-4bb7-b62d-74a9c8fdb4dd"> |
<!--
| FaceTalk_170731_00024_TA/<br>sentence01.wav | ![](doc/video/SAiD/new_model/FaceTalk_170731_00024_TA-sentence01-2-vocaset-diff.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170731_00024_TA-sentence01-1-vrm.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170731_00024_TA-sentence01-3-metahuman.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170731_00024_TA-sentence01-3-unity.mp4){width=200} |
| FaceTalk_170809_00138_TA/<br>sentence02.wav | ![](doc/video/SAiD/new_model/FaceTalk_170809_00138_TA-sentence02-2-vocaset-diff.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170809_00138_TA-sentence02-1-vrm.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170809_00138_TA-sentence02-3-metahuman.mp4){width=200} | ![](doc/video/SAiD/new_model/FaceTalk_170809_00138_TA-sentence02-3-unity.mp4){width=200} |
-->

## Reference

If you use this code as part of any research, please cite the following paper.

```text
```
