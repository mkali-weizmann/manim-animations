# manim-animations
A repository for my manim animations

## How to use:
* For windows, install [chocolatey](https://chocolatey.org/install)
* install manim from [manim documentation](https://docs.manim.community/en/stable/installation.html) (Use the chocolatey option for windows)
* install manim slides from  [manim-slides documentation](https://www.manim.community/plugin/manim-slides/) using:
```bash
pip install manim-slides[manim]
```
 Note: I had to uninstall it and reinstall it for it to work.
```bash
pip install "manim-voiceover[azure,gtts,transcribe]"
```
Note: I first ran `pip install "manim-voiceover[azure,gtts]"` and only then `pip install "manim-voiceover[transcribe]"`, but I guess you can combine them into one command.

install Sox from https://sourceforge.net/projects/sox/

download 2 madlib files and add them to the Sox folder in program files from [here](https://app.box.com/s/tzn5ohyh90viedu3u90w2l2pmp2bl41t) according to the instructions in the main comment in this [stack overflow thread](https://stackoverflow.com/questions/3537155/sox-fail-util-unable-to-load-mad-decoder-library-libmad-function-mad-stream)


Note from manim-voiceover documentation:
Manim needs to be called with the --disable_caching flag due to a bug. Don’t forget to include it every time you render.

## How to make a video file:
Low quality:
```bash
manim -pql file_name.py scene_object_name
```
High quality:
```bash
manim -pqh file_name.py scene_object_name
```

With voiceover (requires to disable caching):
```bash
manim -pql --disable_caching file_name.py scene_object_name
```

