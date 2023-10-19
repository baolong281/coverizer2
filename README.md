# coverizer2

<img width="792" alt="image" src="https://github.com/baolong281/coverizer2/assets/102436898/313a22db-240f-4f57-8786-b27b374e2b02">

forked from [Rothfield](https://huggingface.co/spaces/Rothfeld/stable-diffusion-mat-outpainting-primer)

use ai to expand album covers into wallpapers

## todo

- make ui better
- find out why api requests arent being proxied
- deploy to aws
- use the stable diffusion img2img (need a gpu)
- custom size / aspect ratio

## pipeline [according to creator](https://huggingface.co/spaces/Rothfeld/stable-diffusion-mat-outpainting-primer/discussions/4)

1. use this as primer
2. use outpainting example scripts to improve the filled in spaces

## notes

- inference for img2img on cpu takes a trillion years
