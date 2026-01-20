<h1 align="center">Deep-Live-Cam M1 Optimized</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## Disclaimer

**Personal fork of the Harksider's Deep-Live-Cam repository to apply M1 optimizations.**

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- Ethical Use: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.

- Content Restrictions: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.

- Legal Compliance: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.

- User Responsibility: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## Installation

**1. Set up Your Platform**

- Python (3.10 required)
- pip
- git

**2. Clone the Repository**

```bash
git clone https://github.com/gabriquaranta/deeplivecam-m1
cd deeplivecam-m1
```

**3. Download the Models**

1. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. [inswapper_128_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies**

Apple Silicon (M1/M2/M3) requires specific setup:

```bash
# Install Python 3.10 (specific version is important)
brew install python@3.10

# Install tkinter package (required for the GUI)
brew install python-tk@3.10

# Create and activate virtual environment with Python 3.10
# Install dependencies
./install.sh
```

**Important Notes for macOS:**

- You **must** use Python 3.10, not newer versions like 3.11 or 3.13
- Always run with `python3.10` command not just `python` if you have multiple Python versions installed

## Usage

**1. Image/Video Mode**

- Execute `./run-m1.sh`.
- Choose a source face image and a target image/video.
- Click "Start".
- The output will be saved in a directory named after the target video.

**2. Webcam Mode**

- Execute `./run-m1.sh`.
- Select a source face image.
- Click "Live".
- Wait for the preview to appear (10-30 seconds).
- Use a screen capture tool like OBS to stream.
- To change the face, select a new source image.
