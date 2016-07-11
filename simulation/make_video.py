import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

import subprocess

import filetools


def make_video_from_folder(source_folder, video_filename, fps=10, img_format="%04d.png"):

    video_filename = os.path.abspath(video_filename)

    cmd = 'ffmpeg -r {} -i {} -c:v libx264 -vf fps={} -pix_fmt yuv420p {}'.format(fps, img_format, fps, video_filename)

    if os.path.exists(video_filename):
        os.remove(video_filename)

    p = subprocess.Popen(cmd.split(' '), cwd=source_folder)
    p.wait()


if __name__ == '__main__':

    problem_names = ['circle', 'sinus']
    method_names = ['random', 'uncertainty', 'uncertainty_single']
    # method_names = ['uncertainty', 'uncertainty_single']

    results = {}

    for problem_name in problem_names:
        results[problem_name] = {}
        for method_name in method_names:

            refpath = os.path.join(HERE_PATH, 'plot', problem_name, method_name)

            folders = filetools.list_folders(refpath)
            for subfolders in folders:
                source_folders = filetools.list_folders(subfolders)
                for source_folder in source_folders:
                    print source_folder

                    fname = 'video.avi'
                    video_filename = os.path.join(source_folder, fname)
                    # if not os.path.exists(video_filename):

                    if 'model' in source_folder:
                        fps = 1
                    else:
                        fps = 5

                    make_video_from_folder(source_folder, video_filename, fps=fps, img_format="%06d.png")
